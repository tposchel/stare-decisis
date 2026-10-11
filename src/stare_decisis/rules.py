"""Versioned policy rules: the system's authority claims.

A rule is never edited.  Changing a rule means registering a new version;
every evaluation records the exact `rule_id@version` it applied and who
authored it.

BLOCK rules are tested per option and remove options whose context matches.
MANDATE rules are tested once per decision and, when they fire, force their
target option regardless of expected utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .vocab import AgendaGroup, RuleAction


@dataclass(frozen=True)
class Rule:
    rule_id: str
    version: int
    author: str
    condition: Callable[[dict[str, Any]], bool]
    action: RuleAction = RuleAction.BLOCK
    target_option: str | None = None
    salience: int = 0
    agenda_group: AgendaGroup = AgendaGroup.ENFORCE
    description: str = ""

    @property
    def ref(self) -> str:
        return f"{self.rule_id}@{self.version}"


def split_ref(ref: str) -> tuple[str, int]:
    rid, _, ver = ref.rpartition("@")
    if not rid or not ver.isdigit():
        raise ValueError(f"not a versioned ref: {ref!r}")
    return rid, int(ver)


class RuleBase:
    def __init__(self) -> None:
        self._rules: dict[str, dict[int, Rule]] = {}

    def add(self, rule: Rule) -> Rule:
        if not rule.author:
            raise ValueError("every rule needs an author")
        if rule.action is RuleAction.MANDATE and not rule.target_option:
            raise ValueError("MANDATE rules need a target_option")
        versions = self._rules.setdefault(rule.rule_id, {})
        if rule.version in versions:
            raise ValueError(f"{rule.ref} already registered; rules are immutable, add a new version")
        if versions and rule.version <= max(versions):
            raise ValueError(f"{rule.ref}: versions must increase")
        versions[rule.version] = rule
        return rule

    def __contains__(self, rule_id: str) -> bool:
        return rule_id in self._rules

    def latest(self, rule_id: str) -> Rule:
        versions = self._rules.get(rule_id)
        if not versions:
            raise KeyError(f"unknown rule: {rule_id}")
        return versions[max(versions)]

    def get(self, ref: str) -> Rule:
        rid, ver = split_ref(ref)
        try:
            return self._rules[rid][ver]
        except KeyError:
            raise KeyError(f"unknown rule version: {ref}") from None

    @staticmethod
    def order(rules: list[Rule]) -> list[Rule]:
        return sorted(rules, key=lambda r: (-r.salience, r.rule_id))
