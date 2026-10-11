"""Choice: expected utility under beliefs, filtered and overridden by policy rules.

Pure function so the engine, its dry runs (value-of-information checks), and
the auditor's as-corrected replays all choose the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .decisions import Option, UtilityFunction
from .rules import Rule, RuleBase
from .vocab import ChoiceRule, RuleAction


@dataclass
class Firing:
    rule: Rule
    option_id: str | None
    matched_nodes: list[str]


@dataclass
class Selection:
    selected: str | None
    choice_rule: ChoiceRule
    expected_utility: dict[str, float]
    ranked: list[str]
    blocked: set[str]
    constraint_results: list[dict[str, Any]]
    firings: list[Firing] = field(default_factory=list)
    mandate_conflict: bool = False


def select(
    decision_id: str,
    options: tuple[Option, ...] | list[Option],
    utility: UtilityFunction,
    beliefs: dict[str, dict[str, Any]],
    rules: list[Rule],
    node_view: Callable[[str], dict[str, Any]],
) -> Selection:
    eus: dict[str, float] = {}
    for o in options:
        u = float(utility.fn(o.option_id, beliefs))
        if math.isnan(u):
            raise ValueError(f"{utility.ref} returned NaN for {o.option_id}")
        eus[o.option_id] = u
    ranked = [o.option_id for o in sorted(options, key=lambda o: (-eus[o.option_id], o.option_id))]
    by_id = {o.option_id: o for o in options}
    ordered = RuleBase.order(rules)

    results: list[dict[str, Any]] = []
    firings: list[Firing] = []
    blocked: set[str] = set()

    for oid in ranked:
        opt = by_id[oid]
        ctx = {
            "decision_id": decision_id,
            "option_id": oid,
            "beliefs": beliefs,
            "origin_nodes": [node_view(n) for n in opt.origin_nodes],
        }
        for rule in ordered:
            if rule.action is not RuleAction.BLOCK:
                continue
            fired = bool(rule.condition(ctx))
            if fired:
                blocked.add(oid)
                firings.append(Firing(rule, oid, list(opt.origin_nodes)))
            results.append(
                {
                    "rule": rule.ref,
                    "author": rule.author,
                    "option_id": oid,
                    "matched_nodes": list(opt.origin_nodes) if fired else [],
                    "result": "fail" if fired else "pass",
                }
            )

    mandate_target: str | None = None
    decision_ctx = {"decision_id": decision_id, "option_id": None, "beliefs": beliefs, "origin_nodes": []}
    for rule in ordered:
        if rule.action is not RuleAction.MANDATE:
            continue
        fired = bool(rule.condition(decision_ctx))
        if fired:
            firings.append(Firing(rule, None, []))
            if mandate_target is None:
                mandate_target = rule.target_option
        results.append(
            {
                "rule": rule.ref,
                "author": rule.author,
                "option_id": None,
                "matched_nodes": [],
                "result": "mandate" if fired else "pass",
            }
        )

    if mandate_target is not None:
        if mandate_target not in by_id:
            raise ValueError(f"mandate targets unknown option {mandate_target}")
        conflict = mandate_target in blocked
        return Selection(
            selected=None if conflict else mandate_target,
            choice_rule=ChoiceRule.POLICY_MANDATE,
            expected_utility=eus,
            ranked=ranked,
            blocked=blocked,
            constraint_results=results,
            firings=firings,
            mandate_conflict=conflict,
        )

    selected = next((oid for oid in ranked if oid not in blocked), None)
    return Selection(
        selected=selected,
        choice_rule=ChoiceRule.MAX_EU,
        expected_utility=eus,
        ranked=ranked,
        blocked=blocked,
        constraint_results=results,
        firings=firings,
    )
