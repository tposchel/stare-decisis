"""Audit queries over the log.

Everything except `replay_as_corrected` reads only the log, so a verifier can
run it against a database file with no access to the live engine.  Replay
needs the engine's registries (models, utility versions, rule versions)
because those are code, not data; the log records their hashes and refs so
mismatches are detected.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from .decisions import Option
from .log import AuditLog, ChainCheck, Record
from .selection import select
from .vocab import EventType, NodeType, ReasonCode, ReasoningForm

if TYPE_CHECKING:
    from .engine import Engine


class AuditError(LookupError):
    pass


class Auditor:
    def __init__(self, log: AuditLog, engine: "Engine | None" = None):
        self.log = log
        self.engine = engine

    # -- integrity ---------------------------------------------------------

    def verify(self) -> ChainCheck:
        return self.log.verify()

    # -- lookups -----------------------------------------------------------

    def evaluation(self, decision_id: str, eval_id: str) -> Record:
        for r in self.log.records(EventType.DECISION_EVALUATED):
            if r.payload["decision_id"] == decision_id and r.payload["eval_id"] == eval_id:
                return r
        raise AuditError(f"no evaluation {eval_id} for {decision_id}")

    def evaluations(self, decision_id: str) -> list[Record]:
        return [r for r in self.log.records(EventType.DECISION_EVALUATED) if r.payload["decision_id"] == decision_id]

    def _segment_version(self, ref: str, until_seq: int | None = None) -> Record:
        sid, _, ver = ref.rpartition("@")
        for r in self.log.records(EventType.SEGMENT_VERSION, until_seq=until_seq):
            if r.payload["segment_id"] == sid and r.payload["version"] == int(ver):
                return r
        raise AuditError(f"no segment version {ref}")

    def _node_assertions(self, until_seq: int | None = None) -> dict[str, Record]:
        return {r.payload["node_id"]: r for r in self.log.records(EventType.NODE_ASSERTED, until_seq=until_seq)}

    def _retractions(self, until_seq: int | None = None) -> dict[str, Record]:
        return {r.payload["node_id"]: r for r in self.log.records(EventType.NODE_RETRACTED, until_seq=until_seq)}

    def history(self, decision_id: str) -> list[Record]:
        out = []
        for r in self.log.records():
            p = r.payload
            if p.get("decision_id") == decision_id:
                out.append(r)
        return out

    # -- 1. as made --------------------------------------------------------

    def reconstruct_as_made(self, decision_id: str, eval_id: str) -> dict[str, Any]:
        """The decision exactly as it was made: only records written before it."""
        ev = self.evaluation(decision_id, eval_id)
        segments = {ref: self._segment_version(ref, ev.seq).payload for ref in ev.payload["inputs"]}
        needed: set[str] = set()
        for sp in segments.values():
            needed.update(sp["consumed_nodes"])
        for o in ev.payload["options"]:
            needed.update(o["origin_nodes"])
        asserted = self._node_assertions(ev.seq)
        missing = needed - asserted.keys()
        if missing:
            raise AuditError(f"evaluation references nodes with no prior assertion: {sorted(missing)}")
        known_retractions = self._retractions(ev.seq)
        firings = [
            r.payload
            for r in self.log.records(EventType.RULE_FIRED, until_seq=ev.seq)
            if r.payload["eval_id"] == eval_id
        ]
        return {
            "decision_id": decision_id,
            "eval_id": eval_id,
            "eval_seq": ev.seq,
            "tx_time": ev.tx_time,
            "evaluation": ev.payload,
            "segments": segments,
            "nodes": {nid: {**asserted[nid].payload, "valid_time": asserted[nid].valid_time,
                            "tx_time": asserted[nid].tx_time} for nid in sorted(needed)},
            "retractions_known_at_time": {nid: known_retractions[nid].payload for nid in sorted(needed)
                                          if nid in known_retractions},
            "rule_firings": firings,
        }

    # -- 2. as corrected ---------------------------------------------------

    def replay_as_corrected(
        self,
        decision_id: str,
        eval_id: str,
        as_of_seq: int | None = None,
        reasons: "set[str] | list[str] | None" = None,
    ) -> dict[str, Any]:
        """Re-run an evaluation with every retraction known by `as_of_seq` applied.

        Uses the utility and rule versions the original evaluation recorded, so
        the only thing that changes is the evidence.  `reasons` limits which
        retraction codes count; pass every code except "superseded" to ask
        "was it wrong?" rather than "would it still stand?".
        """
        if self.engine is None:
            raise AuditError("replay needs the engine's model, utility and rule registries")
        eng = self.engine
        made = self.reconstruct_as_made(decision_id, eval_id)
        allowed = None if reasons is None else {str(r) for r in reasons}
        retracted = {
            nid: r for nid, r in self._retractions(as_of_seq).items()
            if allowed is None or r.payload["reason_code"] in allowed
        }
        nodes = made["nodes"]

        beliefs: dict[str, dict[str, Any]] = {}
        excluded: dict[str, list[str]] = {}
        for ref, sp in made["segments"].items():
            sid = sp["segment_id"]
            seg = eng.segments.get(sid)
            if seg is None:
                raise AuditError(f"segment {sid} not registered in engine")
            if seg.model.spec_hash() != sp["model_spec_hash"]:
                raise AuditError(f"model for {sid} differs from the one recorded at {ref}")
            keep = [n for n in sp["consumed_nodes"] if n not in retracted]
            excluded[ref] = [n for n in sp["consumed_nodes"] if n in retracted]
            state = seg.model.fit([seg.extract(nodes[n]["value"]) for n in keep])
            beliefs[sid] = seg.model.summary(state)

        ev = made["evaluation"]
        utility = eng.utilities.get(ev["utility_fn_ref"])
        if utility is None:
            raise AuditError(f"utility {ev['utility_fn_ref']} not registered in engine")
        rule_refs = list(dict.fromkeys(c["rule"] for c in ev["constraints"]))
        rules = [eng.rules.get(ref) for ref in rule_refs]

        options, withdrawn = [], []
        for o in ev["options"]:
            origins = o["origin_nodes"]
            if origins and all(n in retracted for n in origins):
                withdrawn.append(o["option_id"])
                continue
            options.append(Option(o["option_id"], tuple(origins)))

        def view(nid: str) -> dict[str, Any]:
            n = nodes[nid]
            return {"node_id": nid, "node_type": n["node_type"], "reasoning_form": n["reasoning_form"],
                    "value": n["value"], "live": nid not in retracted}

        if options:
            sel = select(decision_id, options, utility, beliefs, rules, view)
            corrected = {"selected": sel.selected, "choice_rule": sel.choice_rule.value,
                         "expected_utility": sel.expected_utility, "blocked": sorted(sel.blocked)}
        else:
            corrected = {"selected": None, "choice_rule": None, "expected_utility": {}, "blocked": []}

        as_made_eu = {o["option_id"]: o["expected_utility"] for o in ev["options"]}
        return {
            "decision_id": decision_id,
            "eval_id": eval_id,
            "as_of_seq": as_of_seq,
            "reasons": sorted(allowed) if allowed is not None else None,
            "as_made": {"selected": ev["selected_option"], "choice_rule": ev["choice_rule"],
                        "expected_utility": as_made_eu},
            "as_corrected": corrected,
            "changed": corrected["selected"] != ev["selected_option"],
            "excluded_nodes": {k: v for k, v in excluded.items() if v},
            "withdrawn_options": withdrawn,
            "corrected_beliefs": beliefs,
        }

    # -- 3. retraction impact ----------------------------------------------

    def retraction_impact(self, node_id: str) -> dict[str, Any]:
        """Every segment version and evaluation the node fed, retracted or not."""
        refs = [
            f"{r.payload['segment_id']}@{r.payload['version']}"
            for r in self.log.records(EventType.SEGMENT_VERSION)
            if node_id in r.payload["consumed_nodes"]
        ]
        ref_set = set(refs)
        evals = []
        for r in self.log.records(EventType.DECISION_EVALUATED):
            p = r.payload
            via = [i for i in p["inputs"] if i in ref_set]
            origin_of = [o["option_id"] for o in p["options"] if node_id in o["origin_nodes"]]
            if via or origin_of:
                evals.append({"decision_id": p["decision_id"], "eval_id": p["eval_id"],
                              "selected_option": p["selected_option"], "via_segments": via,
                              "origin_of_options": origin_of,
                              "selected_depends_on_origin": p["selected_option"] in origin_of})
        retraction = self._retractions().get(node_id)
        return {
            "node_id": node_id,
            "retracted": retraction is not None,
            "retraction": retraction.payload if retraction else None,
            "segment_versions": refs,
            "evaluations": evals,
        }

    # -- 4. retraction clusters --------------------------------------------

    def retraction_clusters(
        self,
        reason_code: ReasonCode | str = ReasonCode.MODEL_HALLUCINATION,
        key: str = "prompt_hash",
    ) -> list[dict[str, Any]]:
        """Group retractions by a provenance field (prompt_hash, model) or source id."""
        reason_code = str(reason_code)
        asserted = self._node_assertions()
        groups: dict[str, list[str]] = defaultdict(list)
        for nid, r in self._retractions().items():
            if r.payload["reason_code"] != reason_code:
                continue
            a = asserted[nid].payload
            if key == "source":
                k = a["source"].get("id")
            else:
                k = (a.get("llm_provenance") or {}).get(key)
            groups[k if k is not None else "<none>"].append(nid)
        return sorted(
            ({"key": k, "count": len(v), "node_ids": sorted(v)} for k, v in groups.items()),
            key=lambda g: (-g["count"], g["key"]),
        )

    # -- 5. reasoning trail ------------------------------------------------

    def reasoning_trail(self, decision_id: str, eval_id: str) -> dict[str, Any]:
        """Which forms of reasoning the selected option rests on."""
        made = self.reconstruct_as_made(decision_id, eval_id)
        ev = made["evaluation"]
        selected = ev["selected_option"]
        origins: list[str] = []
        for o in ev["options"]:
            if o["option_id"] == selected:
                origins = o["origin_nodes"]
        entries = []
        for nid in origins:
            n = made["nodes"][nid]
            entries.append({"node_id": nid, "role": "origin", "node_type": n["node_type"],
                            "reasoning_form": n["reasoning_form"], "source": n["source"].get("id")})
        for ref, sp in made["segments"].items():
            for nid in sp["consumed_nodes"]:
                n = made["nodes"][nid]
                entries.append({"node_id": nid, "role": f"evidence:{ref}", "node_type": n["node_type"],
                                "reasoning_form": n["reasoning_form"], "source": n["source"].get("id")})
        rule_forms = [ReasoningForm.AUTHORITY.value for c in ev["constraints"] if c["result"] != "pass"]

        flags = []
        origin_entries = [e for e in entries if e["role"] == "origin"]
        if origin_entries and all(
            e["node_type"] == NodeType.LLM_HYPOTHESIS.value and e["reasoning_form"] == ReasoningForm.ABDUCTION.value
            for e in origin_entries
        ):
            flags.append("selected_option_rests_on_unverified_llm_abduction")
        evidence_forms = {e["reasoning_form"] for e in entries if e["role"] != "origin"}
        if evidence_forms and evidence_forms <= {ReasoningForm.AUTHORITY.value}:
            flags.append("evidence_is_authority_only")
        if ev["choice_rule"] == "policy_mandate":
            flags.append("choice_by_policy_mandate")
        if ev["choice_rule"] == "human_override":
            flags.append("choice_by_human_override")
        if made["retractions_known_at_time"]:
            flags.append("retracted_nodes_referenced")

        counts = Counter(e["reasoning_form"] for e in entries)
        counts.update(rule_forms)
        return {
            "decision_id": decision_id,
            "eval_id": eval_id,
            "selected_option": selected,
            "choice_rule": ev["choice_rule"],
            "utility_fn_ref": ev["utility_fn_ref"],
            "trail": entries,
            "form_counts": dict(sorted(counts.items())),
            "flags": flags,
        }

    # -- 6. reversal exposure ----------------------------------------------

    def _dependency_state(self, until_seq: int | None):
        published: dict[str, list[str]] = defaultdict(list)
        for nid, r in self._node_assertions(until_seq).items():
            src = r.payload["source"]
            if r.payload["node_type"] == NodeType.DECISION_OUTCOME.value and src.get("decision_id"):
                published[src["decision_id"]].append(nid)
        retracted = set(self._retractions(until_seq))
        latest: dict[str, Record] = {}
        for r in self.log.records(EventType.DECISION_EVALUATED, until_seq=until_seq):
            latest[r.payload["decision_id"]] = r
        consumed = {
            f"{r.payload['segment_id']}@{r.payload['version']}": set(r.payload["consumed_nodes"])
            for r in self.log.records(EventType.SEGMENT_VERSION, until_seq=until_seq)
        }
        return published, retracted, latest, consumed

    @staticmethod
    def _current_dependents(decision_id: str, state) -> dict[str, dict[str, Any]]:
        published, retracted, latest, consumed = state
        out: dict[str, dict[str, Any]] = {}
        frontier = [(decision_id, 0)]
        seen = {decision_id}
        while frontier:
            upstream, depth = frontier.pop(0)
            live = {n for n in published.get(upstream, []) if n not in retracted}
            if not live:
                continue
            for did, ev in sorted(latest.items()):
                if did == decision_id:
                    continue
                p = ev.payload
                evidence = set()
                for ref in p["inputs"]:
                    evidence |= consumed.get(ref, set()) & live
                assumption = set()
                for o in p["options"]:
                    if o["option_id"] == p["selected_option"]:
                        assumption |= set(o["origin_nodes"]) & live
                if not (evidence or assumption):
                    continue
                entry = out.get(did)
                if entry is None:
                    entry = out[did] = {
                        "decision_id": did,
                        "depth": depth + 1,
                        "eval_id": p["eval_id"],
                        "selected_option": p["selected_option"],
                        "via_evidence": [],
                        "via_assumption": [],
                    }
                entry["via_evidence"] = sorted(set(entry["via_evidence"]) | evidence)
                entry["via_assumption"] = sorted(set(entry["via_assumption"]) | assumption)
                if did not in seen:
                    seen.add(did)
                    frontier.append((did, depth + 1))
        return out

    def reversal_exposure(
        self,
        decision_id: str,
        as_of_seq: int | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """What a reversal of this decision would break now, and how that grew.

        A downstream decision is exposed when its standing evaluation uses a
        live published outcome of this decision, either as evidence in a
        segment or as the origin of its selected option, directly or through
        other decisions' published outcomes.  `curve` replays exposure at every
        record that changed it: decision amortization, measured instead of
        modeled.  `weights` assigns a cost per downstream decision (default 1).
        """
        weights = weights or {}
        w = lambda did: float(weights.get(did, 1.0))  # noqa: E731
        state = self._dependency_state(as_of_seq)
        published, retracted, _, _ = state
        current = self._current_dependents(decision_id, state)

        first_pub = min(
            (r.seq for r in self.log.records(EventType.NODE_ASSERTED, until_seq=as_of_seq)
             if r.payload["source"].get("decision_id") == decision_id
             and r.payload["node_type"] == NodeType.DECISION_OUTCOME.value),
            default=None,
        )
        candidate_seqs = [] if first_pub is None else sorted(
            r.seq
            for r in self.log.records(until_seq=as_of_seq)
            if r.seq >= first_pub
            and r.event_type in (EventType.DECISION_EVALUATED.value, EventType.NODE_RETRACTED.value)
            or (r.event_type == EventType.NODE_ASSERTED.value
                and r.payload["node_type"] == NodeType.DECISION_OUTCOME.value)
        )
        curve, last = [], None
        for seq in candidate_seqs:
            deps = self._current_dependents(decision_id, self._dependency_state(seq))
            key = frozenset(deps)
            if key == last:
                continue
            last = key
            curve.append({
                "seq": seq,
                "tx_time": self.log.get(seq).tx_time,
                "dependent_count": len(deps),
                "exposure": sum(w(d) for d in deps),
            })

        dependents = sorted(current.values(), key=lambda e: (e["depth"], e["decision_id"]))
        for e in dependents:
            e["weight"] = w(e["decision_id"])
        return {
            "decision_id": decision_id,
            "as_of_seq": as_of_seq,
            "published_nodes": [
                {"node_id": n, "live": n not in retracted}
                for n in published.get(decision_id, [])
            ],
            "dependents": dependents,
            "dependent_count": len(dependents),
            "max_depth": max((e["depth"] for e in dependents), default=0),
            "exposure": sum(e["weight"] for e in dependents),
            "curve": curve,
        }

    # -- 7. decision amortization ------------------------------------------

    def amortization_schedule(self, decision_id: str) -> dict[str, Any]:
        """The decision's standing against its reversal cost, point by point.

        At each check the log holds regret (how far the standing choice trails
        the best available alternative, in expected-utility units) and R(t)
        (direct reversal cost plus the unwind cost of current dependents).

        status
          sound             regret is zero
          within_tolerance  regret within the decision's voi_threshold
          impaired          regret exceeds tolerance but not R(t): worse than the
                            alternative, yet reversing would cost more (locked in)
          reversed          regret exceeded R(t); the choice changed
        """
        rows: list[dict[str, Any]] = []
        prev_selected = None
        for r in self.log.records():
            p = r.payload
            if p.get("decision_id") != decision_id:
                continue
            if r.event_type == EventType.DECISION_EVALUATED.value:
                sel = p["selected_option"]
                if prev_selected is None:
                    status = "initial"
                elif sel != prev_selected:
                    status = "reversed"
                else:
                    status = "re-evaluated"
                rc = p.get("reversal_cost") or {}
                rows.append({
                    "seq": r.seq, "tx_time": r.tx_time, "event": "evaluated", "eval_id": p["eval_id"],
                    "trigger": p["trigger"], "standing_option": sel, "regret": p["voi_estimate"],
                    "reversal_cost": rc.get("total"), "dependents": sorted(rc.get("dependents", {})),
                    "lock_in_margin": None, "status": status,
                })
                prev_selected = sel
            elif r.event_type == EventType.REEVALUATION_SKIPPED.value:
                rc = p.get("reversal_cost") or {}
                rows.append({
                    "seq": r.seq, "tx_time": r.tx_time, "event": "skipped", "eval_id": p["standing_eval_id"],
                    "trigger": None, "standing_option": p["standing_option"], "regret": p["voi_estimate"],
                    "reversal_cost": rc.get("total"), "dependents": sorted(rc.get("dependents", {})),
                    "lock_in_margin": p.get("lock_in_margin"), "status": p.get("status", "sound"),
                })
            elif r.event_type == EventType.HUMAN_OVERRIDE.value:
                rows.append({
                    "seq": r.seq, "tx_time": r.tx_time, "event": "override", "eval_id": p["eval_id"],
                    "trigger": None, "standing_option": p["chosen_option"], "regret": None,
                    "reversal_cost": None, "dependents": [], "lock_in_margin": None,
                    "status": "reversed" if p["chosen_option"] != p["original_option"] else "re-evaluated",
                })
                prev_selected = p["chosen_option"]
        impaired = [x for x in rows if x["status"] == "impaired"]
        return {
            "decision_id": decision_id,
            "schedule": rows,
            "first_impairment_seq": impaired[0]["seq"] if impaired else None,
            "max_locked_in_regret": max((x["regret"] for x in impaired), default=0.0),
            "reversals": sum(1 for x in rows if x["status"] == "reversed"),
            "currently": rows[-1]["status"] if rows else None,
        }

    # -- 8. skipped re-evaluations -----------------------------------------

    def skipped_reevaluations(self, decision_id: str) -> list[dict[str, Any]]:
        """Answers "why didn't you reconsider when X changed?"."""
        return [r.payload | {"tx_time": r.tx_time}
                for r in self.log.records(EventType.REEVALUATION_SKIPPED)
                if r.payload["decision_id"] == decision_id]
