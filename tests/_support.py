from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stare_decisis import (  # noqa: E402
    AgendaGroup,
    AuditLog,
    BetaBernoulli,
    Decision,
    Engine,
    ManualClock,
    NodeType,
    Option,
    ReasoningForm,
    UtilityFunction,
)

OBS = NodeType.OBSERVATION
IND = ReasoningForm.INDUCTION
SRC = {"id": "test/source"}


def approve_eu(option_id, beliefs, scale=100.0, bar=0.5, seg="rate"):
    if option_id == "deny":
        return 0.0
    return scale * (beliefs[seg]["mean"] - bar)


def utility(version=1, bar=0.5, seg="rate"):
    return UtilityFunction("approve_value", version, "tester",
                           lambda o, b: approve_eu(o, b, bar=bar, seg=seg))


def engine(path: str = ":memory:") -> Engine:
    return Engine(AuditLog(path, clock=ManualClock()))


def simple(path: str = ":memory:", threshold: float = 0.0, origin=()) -> Engine:
    eng = engine(path)
    eng.add_segment("rate", BetaBernoulli(1, 1))
    eng.add_decision(Decision(
        "d1", ("rate",),
        (Option("deny"), Option("approve", origin_nodes=tuple(origin))),
        utility(), voi_threshold=threshold, agenda_group=AgendaGroup.SELECT,
    ))
    return eng


def observe(eng: Engine, node_id: str, value, segment="rate", **kw):
    return eng.assert_node(node_id, OBS, value, IND, SRC, segments=[segment], **kw)


def events(eng: Engine, etype) -> list:
    return eng.log.records(etype)
