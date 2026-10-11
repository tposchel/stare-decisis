"""Decision memory: one decision's path, values, constraints and standing choice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .vocab import AgendaGroup, ChoiceRule


@dataclass(frozen=True)
class Option:
    option_id: str
    origin_nodes: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class UtilityFunction:
    """Expected utility of an option given posterior summaries keyed by segment id.

    Versioned and authored, because utilities are value judgments someone has
    to sign off on.
    """

    fn_id: str
    version: int
    author: str
    fn: Callable[[str, dict[str, dict[str, Any]]], float]
    description: str = ""

    @property
    def ref(self) -> str:
        return f"{self.fn_id}@{self.version}"


@dataclass
class StandingChoice:
    eval_id: str
    inputs: dict[str, int]
    selected: str | None
    choice_rule: ChoiceRule
    expected_utility: dict[str, float]
    seq: int
    override_actor: str | None = None


@dataclass
class Decision:
    decision_id: str
    path: tuple[str, ...]
    options: tuple[Option, ...]
    utility: UtilityFunction
    constraints: tuple[str, ...] = ()
    salience: int = 0
    agenda_group: AgendaGroup = AgendaGroup.SELECT
    voi_threshold: float = 0.0
    direct_reversal_cost: float = 0.0  # cost of undoing this decision itself
    unwind_cost: float = 0.0  # cost of unwinding this decision when something it depends on reverses

    # runtime state (decision memory proper)
    linked_logged: bool = field(default=False, repr=False)
    standing: StandingChoice | None = field(default=None, repr=False)
    checked_versions: dict[str, int] = field(default_factory=dict, repr=False)
    eval_count: int = field(default=0, repr=False)
    force_reason: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = tuple(self.path)
        self.options = tuple(self.options)
        self.constraints = tuple(self.constraints)
        if not self.path:
            raise ValueError("a decision needs at least one segment on its path")
        if not self.options:
            raise ValueError("a decision needs at least one option")
        ids = [o.option_id for o in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("option ids must be unique")
        if self.voi_threshold < 0:
            raise ValueError("voi_threshold must be non-negative")
        if self.direct_reversal_cost < 0 or self.unwind_cost < 0:
            raise ValueError("reversal and unwind costs must be non-negative")

    def option(self, option_id: str) -> Option:
        for o in self.options:
            if o.option_id == option_id:
                return o
        raise KeyError(f"{self.decision_id}: unknown option {option_id}")


@dataclass(frozen=True)
class Outcome:
    decision_id: str
    kind: str  # evaluated | skipped | unlinked | unchanged
    eval_id: str | None = None
    selected: str | None = None
    voi_estimate: float | None = None
