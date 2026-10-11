"""Walkthrough: a credit-line-increase decision from evidence to audit.

    python examples/credit_line.py [path/to/ledger.db]

Shows linking, lazy evaluation, a skipped re-evaluation with its regret
figure, a hallucinated LLM hypothesis being retracted, and both
reconstructions: as made, and as corrected.
"""

from __future__ import annotations

import json
import sys

from stare_decisis import (
    AgendaGroup,
    AuditLog,
    BetaBernoulli,
    Decision,
    Engine,
    GaussianMixtureEM,
    LLMProvenance,
    ManualClock,
    NodeType,
    NormalKnownVariance,
    Option,
    ReasonCode,
    ReasoningForm,
    Rule,
    UtilityFunction,
)

OBS, LLM = NodeType.OBSERVATION, NodeType.LLM_HYPOTHESIS
IND, ABD, AUTH = ReasoningForm.INDUCTION, ReasoningForm.ABDUCTION, ReasoningForm.AUTHORITY


def show(title: str, obj) -> None:
    print(f"\n== {title}")
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def build(path: str = ":memory:") -> Engine:
    eng = Engine(AuditLog(path, clock=ManualClock("2026-10-01T09:00:00+00:00")))

    eng.add_segment("repayment", BetaBernoulli(2, 2), prior_ref="policy/priors/repayment@1")
    eng.add_segment("income_growth", NormalKnownVariance(0.0, 0.01, 0.0025), prior_ref="policy/priors/income@1")
    eng.add_segment("spend_regime", GaussianMixtureEM(k=2))

    eng.add_rule(Rule(
        "min_repayment", 1, author="Credit Policy Committee",
        condition=lambda c: c["option_id"] == "increase" and c["beliefs"]["repayment"]["mean"] < 0.75,
        description="No increase when expected repayment is below 75%",
    ))

    def eu(option_id, b):
        if option_id == "hold":
            return 0.0
        repay = b["repayment"]["mean"]
        growth = b["income_growth"]["mean"]
        high_spend = b["spend_regime"]["p_high"] or 0.0
        return 1000 * (repay - 0.8) + 20000 * growth - 300 * high_spend

    eng.add_decision(Decision(
        decision_id="cli-4471",
        path=("repayment", "income_growth", "spend_regime"),
        options=(Option("hold"), Option("increase", origin_nodes=("h-newjob",))),
        utility=UtilityFunction("cli_value", 1, author="Risk Analytics", fn=eu,
                                description="Expected margin of an increase, in dollars"),
        constraints=("min_repayment",),
        agenda_group=AgendaGroup.SELECT,
        voi_threshold=5.0,
    ))
    return eng


def main(path: str = ":memory:") -> None:
    eng = build(path)
    src = {"id": "bureau/feed-7"}

    for i, paid in enumerate([1, 1, 1, 1, 0, 1, 1, 1, 1, 1]):
        eng.assert_node(f"pay-{i}", OBS, paid, IND, src, valid_time=f"2026-0{1 + i % 9}-15", segments=["repayment"])
    for i, spend in enumerate([310, 295, 320, 305, 880, 900, 300, 315]):
        eng.assert_node(f"spend-{i}", OBS, spend, IND, {"id": "ledger/cards"}, segments=["spend_regime"])

    eng.assert_node("inc-1", OBS, 0.02, IND, {"id": "payroll/extract"}, segments=["income_growth"])
    eng.assert_node(
        "h-newjob", LLM,
        {"claim": "Applicant changed employer in August; salary up ~15%", "basis": "free-text note"},
        ABD, {"id": "llm/underwriting-assist"},
        llm_provenance=LLMProvenance("assist-v3", "sha256:prompt-income-explainer", temperature=0.2, seed=7),
    )
    eng.assert_node("inc-2", OBS, 0.15, IND, {"id": "llm/underwriting-assist"}, segments=["income_growth"])

    show("first run", [o.__dict__ for o in eng.run()])

    eng.assert_node("pay-10", OBS, 1, IND, src, segments=["repayment"])
    show("one more on-time payment (should be skipped)", [o.__dict__ for o in eng.run()])

    eng.retract("h-newjob", ReasonCode.MODEL_HALLUCINATION,
                "No employer change in HR verification; note referred to spouse",
                discovered_by="analyst:jlee", discovery_ref="ticket/UW-2291")
    eng.retract("inc-2", ReasonCode.EXTRACTION_ERROR,
                "Growth figure derived from the retracted hypothesis",
                discovered_by="analyst:jlee", discovery_ref="ticket/UW-2291")
    show("after retractions", [o.__dict__ for o in eng.run()])

    audit = eng.auditor()
    show("chain", audit.verify().__dict__)
    show("trail for cli-4471#1", audit.reasoning_trail("cli-4471", "cli-4471#1"))
    show("as corrected (cli-4471#1)", audit.replay_as_corrected("cli-4471", "cli-4471#1"))
    show("impact of h-newjob", audit.retraction_impact("h-newjob"))
    show("hallucination clusters", audit.retraction_clusters())
    show("skipped re-evaluations", audit.skipped_reevaluations("cli-4471"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ":memory:")
