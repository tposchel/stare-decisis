"""The engine: three memory layers, a lazy goal queue, and the audit log.

Work happens in two phases, as in PHREAK:

1. assert / retract: append to the log, stage the change in the affected
   segments, update linking bits.  No belief computation.
2. run: pop linked, dirty decisions from the goal queue (agenda group, then
   salience), flush only the segments on that decision's path, then either
   evaluate the decision or log why re-evaluation was not worth it.

Dirtiness is graded, not binary: a changed segment triggers re-evaluation only
when the staleness regret of the standing choice exceeds the decision's
`voi_threshold`.  Staleness regret is the expected-utility gap, under the new
beliefs, between the choice the system would now make and the choice that
stands.  It is a value-of-information proxy: zero when the new evidence
cannot change the choice.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any, Iterable

from .clock import Clock, parse_instant
from .decisions import Decision, Outcome, StandingChoice, UtilityFunction
from .hashing import content_hash
from .log import AuditLog, Record
from .models import SegmentModel
from .nodes import LLMProvenance, Node, NodeMemory, Retraction
from .rules import Rule, RuleBase
from .segments import Segment, SegmentVersion
from .selection import Selection, select
from .vocab import (
    AGENDA_ORDER,
    AgendaGroup,
    ChoiceRule,
    EventType,
    NodeType,
    ReasonCode,
    ReasoningForm,
)


class Engine:
    def __init__(self, log: AuditLog | None = None, clock: Clock | None = None):
        if log is not None and clock is not None:
            raise ValueError("pass a clock to the AuditLog, not to the Engine, when supplying a log")
        self.log = log if log is not None else AuditLog(clock=clock)
        self.nodes = NodeMemory()
        self.segments: dict[str, Segment] = {}
        self.rules = RuleBase()
        self.decisions: dict[str, Decision] = {}
        self.utilities: dict[str, UtilityFunction] = {}
        self._routes: dict[str, list[str]] = defaultdict(list)  # node -> segments
        self._node_versions: dict[str, list[str]] = defaultdict(list)  # node -> segment refs
        self._published: dict[str, list[str]] = defaultdict(list)  # decision -> outcome nodes

    # -- registration ------------------------------------------------------

    def add_segment(
        self,
        segment_id: str,
        model: SegmentModel,
        prior_ref: str | None = None,
        extract=None,
        freshness: timedelta | None = None,
    ) -> Segment:
        if segment_id in self.segments:
            raise ValueError(f"segment already exists: {segment_id}")
        seg = Segment(segment_id, model, prior_ref=prior_ref, extract=extract, freshness=freshness)
        self.segments[segment_id] = seg
        return seg

    def add_rule(self, rule: Rule) -> Rule:
        return self.rules.add(rule)

    def register_utility(self, utility: UtilityFunction) -> UtilityFunction:
        if not utility.author:
            raise ValueError("every utility function needs an author")
        existing = self.utilities.get(utility.ref)
        if existing is not None and existing is not utility:
            raise ValueError(f"{utility.ref} already registered; add a new version instead")
        self.utilities[utility.ref] = utility
        return utility

    def add_decision(self, decision: Decision) -> Decision:
        if decision.decision_id in self.decisions:
            raise ValueError(f"decision already exists: {decision.decision_id}")
        for s in decision.path:
            if s not in self.segments:
                raise KeyError(f"{decision.decision_id}: unknown segment {s}")
        for r in decision.constraints:
            if r not in self.rules:
                raise KeyError(f"{decision.decision_id}: unknown rule {r}")
        self.register_utility(decision.utility)
        self.decisions[decision.decision_id] = decision
        self._update_linking()
        return decision

    def set_utility(self, decision_id: str, utility: UtilityFunction) -> None:
        """Swap in a new utility version.  Marks the decision for re-evaluation."""
        d = self.decisions[decision_id]
        self.register_utility(utility)
        d.utility = utility
        d.force_reason = "utility_changed"

    # -- phase 1: assertion and retraction (staging only) ---------------------

    def assert_node(
        self,
        node_id: str,
        node_type: NodeType,
        value: Any,
        reasoning_form: ReasoningForm,
        source: dict[str, Any],
        *,
        valid_time: str | None = None,
        llm_provenance: LLMProvenance | None = None,
        segments: Iterable[str] = (),
    ) -> Node:
        node_type = NodeType(node_type)
        reasoning_form = ReasoningForm(reasoning_form)
        segments = list(segments)
        if node_type is NodeType.LLM_HYPOTHESIS and llm_provenance is None:
            raise ValueError("llm_hypothesis nodes require llm_provenance")
        if llm_provenance is not None:
            frozen = self.nodes.frozen_for(llm_provenance)
            if frozen is not None:
                return frozen  # output of record; never regenerated
        if node_id in self.nodes:
            raise ValueError(f"node already exists: {node_id}")
        if not source or "id" not in source:
            raise ValueError("source must include an 'id'")
        for s in segments:
            if s not in self.segments:
                raise KeyError(f"unknown segment: {s}")

        value_hash = content_hash(value)
        provenance = None
        if llm_provenance is not None:
            provenance = {
                "model": llm_provenance.model,
                "prompt_hash": llm_provenance.prompt_hash,
                "output_hash": value_hash,
                "temperature": llm_provenance.temperature,
                "seed": llm_provenance.seed,
            }
        rec = self.log.append(
            EventType.NODE_ASSERTED,
            {
                "node_id": node_id,
                "node_type": node_type,
                "value": value,
                "value_hash": value_hash,
                "reasoning_form": reasoning_form,
                "source": source,
                "llm_provenance": provenance,
                "segments": segments,
            },
            valid_time=valid_time,
        )
        node = Node(
            node_id=node_id,
            node_type=node_type,
            value=rec.payload["value"],
            value_hash=value_hash,
            reasoning_form=reasoning_form,
            source=rec.payload["source"],
            valid_time=valid_time,
            tx_time=rec.tx_time,
            seq=rec.seq,
            llm_provenance=llm_provenance,
        )
        self.nodes.add(node)
        for s in segments:
            self.segments[s].stage_insert(node_id)
            self._routes[node_id].append(s)
        for d in self.decisions.values():
            if d.standing and any(node_id in o.origin_nodes for o in d.options):
                d.force_reason = d.force_reason or "option_origin_asserted"
        self._update_linking()
        return node

    def retract(
        self,
        node_id: str,
        reason_code: ReasonCode,
        reason_text: str,
        discovered_by: str,
        discovery_ref: str | None = None,
    ) -> Record:
        node = self.nodes.get(node_id)
        if not node.live:
            raise ValueError(f"node already retracted: {node_id}")
        reason_code = ReasonCode(reason_code)
        rec = self.log.append(
            EventType.NODE_RETRACTED,
            {
                "node_id": node_id,
                "reason_code": reason_code,
                "reason_text": reason_text,
                "discovered_by": discovered_by,
                "discovery_ref": discovery_ref,
                "affected_segments": sorted(self._node_versions.get(node_id, [])),
                "pending_segments": [
                    s for s in self._routes.get(node_id, []) if s not in {r.split("@")[0] for r in self._node_versions.get(node_id, [])}
                ],
            },
        )
        self.nodes.mark_retracted(
            node_id,
            Retraction(rec.tx_time, reason_code, reason_text, discovered_by, discovery_ref, rec.seq),
        )
        for s in self._routes.get(node_id, []):
            self.segments[s].stage_delete(node_id)
        for d in self.decisions.values():
            if d.standing and any(node_id in o.origin_nodes for o in d.options):
                d.force_reason = "option_origin_retracted"
        return rec

    def expire(self, as_of: str) -> list[Record]:
        """Stage removal of evidence older than each segment's freshness window.

        Age is measured from the node's valid_time (when it was true), falling
        back to tx_time when no valid_time was given.  Expiry is scheduled
        housekeeping, not a correction: as-corrected replays do not exclude
        expired nodes, because they were sound evidence when used.
        """
        now = parse_instant(as_of)
        by_node: dict[str, dict[str, float]] = defaultdict(dict)
        for seg in self.segments.values():
            if seg.freshness is None:
                continue
            for nid in seg.projected_members():
                n = self.nodes.get(nid)
                if now - parse_instant(n.valid_time or n.tx_time) > seg.freshness:
                    by_node[nid][seg.segment_id] = seg.freshness.total_seconds()
        records = []
        for nid in sorted(by_node):
            n = self.nodes.get(nid)
            windows = by_node[nid]
            records.append(self.log.append(
                EventType.NODE_EXPIRED,
                {
                    "node_id": nid,
                    "as_of": as_of,
                    "aged_from": "valid_time" if n.valid_time else "tx_time",
                    "segments": sorted(windows),
                    "freshness_seconds": windows,
                    "affected_segments": sorted(
                        r for r in self._node_versions.get(nid, []) if r.split("@")[0] in windows
                    ),
                },
            ))
            for sid in windows:
                self.segments[sid].stage_expire(nid)
        return records

    def _update_linking(self) -> None:
        for d in self.decisions.values():
            if d.linked_logged:
                continue
            if all(self.segments[s].linked for s in d.path):
                self.log.append(
                    EventType.DECISION_LINKED,
                    {
                        "decision_id": d.decision_id,
                        "path": [f"{s}@{self.segments[s].version}" for s in d.path],
                        "staged": {s: self.segments[s].dirty for s in d.path},
                    },
                )
                d.linked_logged = True

    # -- phase 2: lazy evaluation ------------------------------------------

    def _is_goal(self, d: Decision) -> bool:
        if not all(self.segments[s].linked for s in d.path):
            # an evaluated decision whose evidence has drained still needs one
            # pass so the drain is versioned and the unlink is logged
            return d.standing is not None and d.linked_logged and any(self.segments[s].dirty for s in d.path)
        if d.standing is None or d.force_reason:
            return True
        if any(self.segments[s].dirty for s in d.path):
            return True
        return any(self.segments[s].version != d.checked_versions.get(s) for s in d.path)

    def pending(self, agenda_group: AgendaGroup | None = None) -> list[Decision]:
        goals = [
            d
            for d in self.decisions.values()
            if (agenda_group is None or d.agenda_group is agenda_group) and self._is_goal(d)
        ]
        return sorted(goals, key=lambda d: (-d.salience, d.decision_id))

    def run(self, agenda_group: AgendaGroup | None = None, max_passes: int = 100) -> list[Outcome]:
        """Process pending decisions.

        With an agenda group, only that group is processed (lazy).  Without
        one, every group is processed in stage order, and the passes repeat
        until nothing is pending, so a reversal that cascades into an earlier
        group still settles within the same call.
        """
        outcomes: list[Outcome] = []
        if agenda_group is not None:
            self._drain(AgendaGroup(agenda_group), outcomes)
            return outcomes
        for _ in range(max_passes):
            for g in AGENDA_ORDER:
                self._drain(g, outcomes)
            if not self.pending():
                return outcomes
        raise RuntimeError(
            f"decision cascade did not settle after {max_passes} passes;"
            " check for cycles among published decision outcomes"
        )

    def _drain(self, group: AgendaGroup, outcomes: list[Outcome]) -> None:
        while True:
            goals = self.pending(group)
            if not goals:
                return
            outcomes.append(self._process(goals[0]))

    # -- decision dependencies ---------------------------------------------

    def publish_outcome(
        self,
        decision_id: str,
        node_id: str,
        *,
        value: Any = None,
        segments: Iterable[str] = (),
        reasoning_form: ReasoningForm = ReasoningForm.AUTHORITY,
        valid_time: str | None = None,
    ) -> Node:
        """Turn a decision's standing choice into a node other decisions can build on.

        The node is contingent: it records which choice it assumes.  If the
        decision later settles on a different choice (re-evaluation or human
        override), the node is retracted as `superseded` and the change
        cascades to every decision that consumed it.  Use it as an option's
        origin ("this option assumes X holds") or route it into segments.
        """
        d = self.decisions[decision_id]
        st = d.standing
        if st is None or st.selected is None:
            raise ValueError(f"{decision_id} has no standing choice to publish")
        source = {
            "id": f"decision:{decision_id}",
            "decision_id": decision_id,
            "eval_id": st.eval_id,
            "selected_option": st.selected,
            "choice_rule": st.choice_rule.value,
        }
        if value is None:
            value = {"decision_id": decision_id, "selected_option": st.selected}
        node = self.assert_node(
            node_id, NodeType.DECISION_OUTCOME, value, reasoning_form, source,
            valid_time=valid_time, segments=segments,
        )
        self._published[decision_id].append(node_id)
        return node

    def dependents(self, decision_id: str) -> dict[str, int]:
        """Decisions whose standing choice rests on this one's live published
        outcomes, directly or transitively, mapped to their depth."""
        out: dict[str, int] = {}
        frontier, seen = [(decision_id, 0)], {decision_id}
        while frontier:
            upstream, depth = frontier.pop(0)
            live = {n for n in self._published.get(upstream, []) if self.nodes.get(n).live}
            if not live:
                continue
            for did, d in sorted(self.decisions.items()):
                if did == decision_id or d.standing is None:
                    continue
                used = any(
                    live & set(self.segments[s].versions[v].consumed_nodes)
                    for s, v in d.standing.inputs.items()
                    if v in self.segments[s].versions
                )
                if not used and d.standing.selected is not None:
                    used = bool(live & set(d.option(d.standing.selected).origin_nodes))
                if not used:
                    continue
                out.setdefault(did, depth + 1)
                if did not in seen:
                    seen.add(did)
                    frontier.append((did, depth + 1))
        return out

    def reversal_cost(self, decision_id: str) -> dict[str, Any]:
        """R(t): the direct cost of undoing the decision plus the unwind cost of
        every decision currently resting on it.  In expected-utility units."""
        d = self.decisions[decision_id]
        deps = self.dependents(decision_id)
        exposure = sum(self.decisions[x].unwind_cost for x in deps)
        return {
            "direct": d.direct_reversal_cost,
            "exposure": exposure,
            "total": d.direct_reversal_cost + exposure,
            "dependents": {x: deps[x] for x in sorted(deps)},
        }

    def _cascade(self, decision_id: str, new_selected: str | None, ref: str) -> list[Record]:
        out = []
        for nid in self._published.get(decision_id, []):
            n = self.nodes.get(nid)
            assumed = n.source.get("selected_option")
            if n.live and assumed != new_selected:
                out.append(self.retract(
                    nid, ReasonCode.SUPERSEDED,
                    f"{decision_id} changed from {assumed} to {new_selected}",
                    discovered_by="engine:cascade", discovery_ref=ref,
                ))
        return out

    def _flush(self, segment_id: str) -> SegmentVersion | None:
        seg = self.segments[segment_id]
        v = seg.flush(lambda nid: self.nodes.get(nid).value)
        if v is None:
            return None
        self.log.append(
            EventType.SEGMENT_VERSION,
            {
                "segment_id": v.segment_id,
                "version": v.version,
                "consumed_nodes": list(v.consumed_nodes),
                "inserted_nodes": list(v.inserted_nodes),
                "retracted_nodes": list(v.retracted_nodes),
                "expired_nodes": list(v.expired_nodes),
                "model_spec": seg.model.spec(),
                "model_spec_hash": v.model_spec_hash,
                "sufficient_stats": v.state,
                "sufficient_stats_hash": v.sufficient_stats_hash,
                "summary": v.summary,
                "prior_ref": v.prior_ref,
                "recompute_method": v.recompute_method,
                "trigger": v.trigger,
            },
        )
        for nid in v.consumed_nodes:
            self._node_versions[nid].append(v.ref)
        return v

    def _beliefs(self, d: Decision) -> dict[str, dict[str, Any]]:
        return {s: self.segments[s].summary for s in d.path}

    def _node_view(self, node_id: str) -> dict[str, Any]:
        if node_id not in self.nodes:
            return {"node_id": node_id, "node_type": None, "reasoning_form": None,
                    "value": None, "live": False, "asserted": False}
        n = self.nodes.get(node_id)
        return {
            "node_id": n.node_id,
            "node_type": n.node_type.value,
            "reasoning_form": n.reasoning_form.value,
            "value": n.value,
            "live": n.live,
        }

    def _rules_for(self, d: Decision) -> list[Rule]:
        return [self.rules.latest(r) for r in d.constraints]

    def _option_status(self, o) -> str:
        """available | pending (no origin asserted yet) | withdrawn (every asserted origin retracted)."""
        if not o.origin_nodes:
            return "available"
        known = [self.nodes.get(n) for n in o.origin_nodes if n in self.nodes]
        if not known:
            return "pending"
        return "available" if any(n.live for n in known) else "withdrawn"

    def _live_options(self, d: Decision):
        return tuple(o for o in d.options if self._option_status(o) == "available")

    def _process(self, d: Decision) -> Outcome:
        for s in d.path:
            self._flush(s)
        versions = {s: self.segments[s].version for s in d.path}

        empty = [s for s in d.path if not self.segments[s].consumed]
        if empty:
            self.log.append(
                EventType.DECISION_UNLINKED,
                {
                    "decision_id": d.decision_id,
                    "empty_segments": empty,
                    "standing_eval_id": d.standing.eval_id if d.standing else None,
                    "standing_option": d.standing.selected if d.standing else None,
                },
            )
            d.linked_logged = False
            d.checked_versions = versions
            return Outcome(d.decision_id, "unlinked")

        beliefs = self._beliefs(d)
        rules = self._rules_for(d)
        options = self._live_options(d)

        if d.standing is None:
            return self._evaluate(d, versions, beliefs, rules, options, trigger="initial")

        if not options:
            return self._evaluate(d, versions, beliefs, rules, options, trigger="all_options_withdrawn")
        if d.force_reason:
            return self._evaluate(d, versions, beliefs, rules, options, trigger=d.force_reason)

        dry = select(d.decision_id, options, d.utility, beliefs, rules, self._node_view)
        old = d.standing.selected
        changed_refs = [f"{s}@{versions[s]}" for s in d.path if d.standing.inputs.get(s) != versions[s]]

        if old is None or old not in dry.expected_utility or old in dry.blocked or dry.selected is None:
            return self._evaluate(d, versions, beliefs, rules, options, trigger="standing_choice_invalid", dry=dry)
        if dry.choice_rule is ChoiceRule.POLICY_MANDATE and dry.selected != old:
            return self._evaluate(d, versions, beliefs, rules, options, trigger="mandate", dry=dry)

        regret = 0.0 if dry.selected == old else dry.expected_utility[dry.selected] - dry.expected_utility[old]
        regret = max(regret, 0.0)
        rc = self.reversal_cost(d.decision_id)
        hurdle = d.voi_threshold + rc["total"]
        if regret <= hurdle:
            status = (
                "sound" if regret == 0
                else "within_tolerance" if regret <= d.voi_threshold
                else "impaired"
            )
            self.log.append(
                EventType.REEVALUATION_SKIPPED,
                {
                    "decision_id": d.decision_id,
                    "standing_eval_id": d.standing.eval_id,
                    "standing_option": old,
                    "trigger_segments": changed_refs,
                    "voi_method": "staleness_regret",
                    "voi_estimate": regret,
                    "voi_threshold": d.voi_threshold,
                    "could_flip": dry.selected != old,
                    "would_select": dry.selected,
                    "reversal_cost": rc,
                    "hurdle": hurdle,
                    "lock_in_margin": hurdle - regret,
                    "status": status,
                },
            )
            d.checked_versions = versions
            return Outcome(d.decision_id, "skipped", d.standing.eval_id, old, regret)

        return self._evaluate(d, versions, beliefs, rules, options, trigger="voi_exceeded",
                              dry=dry, regret=regret, reversal=rc)

    def _evaluate(
        self,
        d: Decision,
        versions: dict[str, int],
        beliefs: dict[str, dict[str, Any]],
        rules: list[Rule],
        options,
        *,
        trigger: str,
        dry: Selection | None = None,
        regret: float | None = None,
        reversal: dict[str, Any] | None = None,
    ) -> Outcome:
        d.eval_count += 1
        eval_id = f"{d.decision_id}#{d.eval_count}"

        if options:
            sel = select(d.decision_id, options, d.utility, beliefs, rules, self._node_view)
        else:
            sel = Selection(None, ChoiceRule.MAX_EU, {}, [], set(), [])

        for f in sel.firings:
            self.log.append(
                EventType.RULE_FIRED,
                {
                    "rule": f.rule.ref,
                    "author": f.rule.author,
                    "action": f.rule.action,
                    "agenda_group": f.rule.agenda_group,
                    "decision_id": d.decision_id,
                    "eval_id": eval_id,
                    "option_id": f.option_id,
                    "matched_nodes": f.matched_nodes,
                },
            )

        selected, choice_rule, actor = sel.selected, sel.choice_rule, "system"
        system_recommendation = None
        flags: list[str] = []
        prior = d.standing
        if prior is not None and prior.choice_rule is ChoiceRule.HUMAN_OVERRIDE:
            human_choice_valid = (
                prior.selected in sel.expected_utility
                and prior.selected not in sel.blocked
                and sel.choice_rule is not ChoiceRule.POLICY_MANDATE
            )
            if human_choice_valid:
                system_recommendation = sel.selected
                selected, choice_rule, actor = prior.selected, ChoiceRule.HUMAN_OVERRIDE, prior.override_actor or "human"
                flags.append("human_override_stands")
            else:
                flags.append("human_override_revoked_by_policy")
        if sel.mandate_conflict:
            flags.append("mandate_conflicts_with_block")
        if not options:
            flags.append("all_options_withdrawn")

        withdrawn = [o.option_id for o in d.options if self._option_status(o) == "withdrawn"]
        pending_opts = [o.option_id for o in d.options if self._option_status(o) == "pending"]
        payload = {
            "decision_id": d.decision_id,
            "eval_id": eval_id,
            "agenda_group": d.agenda_group,
            "salience": d.salience,
            "inputs": [f"{s}@{versions[s]}" for s in d.path],
            "beliefs": {f"{s}@{versions[s]}": beliefs[s] for s in d.path},
            "options": [
                {
                    "option_id": o.option_id,
                    "origin_nodes": list(o.origin_nodes),
                    "expected_utility": sel.expected_utility.get(o.option_id),
                    "blocked": o.option_id in sel.blocked,
                }
                for o in options
            ],
            "withdrawn_options": withdrawn,
            "pending_options": pending_opts,
            "utility_fn_ref": d.utility.ref,
            "utility_fn_author": d.utility.author,
            "constraints": sel.constraint_results,
            "selected_option": selected,
            "choice_rule": choice_rule,
            "system_recommendation": system_recommendation,
            "actor": actor,
            "trigger": trigger,
            "voi_estimate": regret,
            "reversal_cost": reversal,
            "direct_reversal_cost": d.direct_reversal_cost,
            "unwind_cost": d.unwind_cost,
            "supersedes": prior.eval_id if prior else None,
            "flags": flags,
        }
        rec = self.log.append(EventType.DECISION_EVALUATED, payload)
        d.standing = StandingChoice(
            eval_id=eval_id,
            inputs=dict(versions),
            selected=selected,
            choice_rule=choice_rule,
            expected_utility=dict(sel.expected_utility),
            seq=rec.seq,
            override_actor=prior.override_actor if choice_rule is ChoiceRule.HUMAN_OVERRIDE and prior else None,
        )
        d.checked_versions = dict(versions)
        d.force_reason = None
        self._cascade(d.decision_id, selected, eval_id)
        return Outcome(d.decision_id, "evaluated", eval_id, selected, regret)

    # -- humans ------------------------------------------------------------

    def human_override(
        self,
        decision_id: str,
        chosen_option: str,
        actor: str,
        reason_code: str,
        reason_text: str,
    ) -> Record:
        d = self.decisions[decision_id]
        if d.standing is None:
            raise ValueError(f"{decision_id} has not been evaluated; nothing to override")
        d.option(chosen_option)
        if not actor:
            raise ValueError("override needs an actor")
        rec = self.log.append(
            EventType.HUMAN_OVERRIDE,
            {
                "decision_id": decision_id,
                "eval_id": d.standing.eval_id,
                "actor": actor,
                "original_option": d.standing.selected,
                "chosen_option": chosen_option,
                "reason_code": reason_code,
                "reason_text": reason_text,
            },
        )
        d.standing.selected = chosen_option
        d.standing.choice_rule = ChoiceRule.HUMAN_OVERRIDE
        d.standing.override_actor = actor
        self._cascade(decision_id, chosen_option, f"{d.standing.eval_id}/override@{rec.seq}")
        return rec

    # -- convenience -------------------------------------------------------

    def standing(self, decision_id: str) -> StandingChoice | None:
        return self.decisions[decision_id].standing

    def auditor(self):
        from .audit import Auditor

        return Auditor(self.log, self)
