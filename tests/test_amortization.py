import unittest

from _support import IND, OBS, SRC, engine, events

from stare_decisis import AgendaGroup, BetaBernoulli, Decision, EventType, Option, UtilityFunction


def build(direct=5.0, unwind_integrate=10.0, unwind_launch=15.0):
    """vendor -> integrate (assumption) -> launch (evidence).  R(vendor) = 5 + 10 + 15 = 30."""
    eng = engine()
    eng.add_segment("fit", BetaBernoulli())
    eng.add_segment("capacity", BetaBernoulli())
    eng.add_segment("readiness", BetaBernoulli(), extract=lambda v: 1)
    eng.add_decision(Decision(
        "vendor", ("fit",), (Option("drop"), Option("keep")),
        UtilityFunction("vendor_u", 1, "proc", lambda o, b: 100 * (b["fit"]["mean"] - 0.5) if o == "keep" else 0.0),
        direct_reversal_cost=direct,
    ))
    eng.add_decision(Decision(
        "integrate", ("capacity",), (Option("wait"), Option("build", origin_nodes=("vendor-kept",))),
        UtilityFunction("integrate_u", 1, "eng", lambda o, b: 10.0 if o == "build" else 0.0),
        unwind_cost=unwind_integrate,
    ))
    eng.add_decision(Decision(
        "launch", ("readiness",), (Option("hold"), Option("go")),
        UtilityFunction("launch_u", 1, "pm", lambda o, b: 100 * (b["readiness"]["mean"] - 0.5) if o == "go" else 0.0),
        unwind_cost=unwind_launch, agenda_group=AgendaGroup.SCORE,
    ))
    eng.assert_node("fit-1", OBS, 1, IND, SRC, segments=["fit"])
    eng.assert_node("cap-1", OBS, 1, IND, SRC, segments=["capacity"])
    eng.run()
    eng.publish_outcome("vendor", "vendor-kept")
    eng.run()
    eng.publish_outcome("integrate", "integrate-built", segments=["readiness"])
    eng.run()
    return eng


def bad_news(eng, n, start=0):
    for i in range(start, start + n):
        eng.assert_node(f"fit-bad-{i}", OBS, 0, IND, SRC, segments=["fit"])
    return eng.run()


class ReversalCost(unittest.TestCase):
    def test_measured_from_dependents(self):
        rc = build().reversal_cost("vendor")
        self.assertEqual(rc["dependents"], {"integrate": 1, "launch": 2})
        self.assertEqual((rc["direct"], rc["exposure"], rc["total"]), (5.0, 25.0, 30.0))

    def test_grows_as_decisions_build_on_it(self):
        eng = build()
        self.assertEqual(eng.reversal_cost("integrate")["total"], 15.0)
        self.assertEqual(eng.reversal_cost("launch")["total"], 0.0)

    def test_negative_costs_rejected(self):
        with self.assertRaises(ValueError):
            Decision("x", ("s",), (Option("a"),), UtilityFunction("u", 1, "a", lambda o, b: 0), unwind_cost=-1)


class LockIn(unittest.TestCase):
    def test_impaired_then_reversed(self):
        eng = build()
        bad_news(eng, 2)  # mean 2/5: keep trails drop by 10 < R = 30
        self.assertEqual(eng.standing("vendor").selected, "keep")
        skip = events(eng, EventType.REEVALUATION_SKIPPED)[-1].payload
        self.assertEqual(skip["status"], "impaired")
        self.assertAlmostEqual(skip["voi_estimate"], 10.0)
        self.assertEqual((skip["hurdle"], skip["reversal_cost"]["total"]), (30.0, 30.0))
        self.assertAlmostEqual(skip["lock_in_margin"], 20.0)
        self.assertEqual(events(eng, EventType.NODE_RETRACTED), [])

        bad_news(eng, 8, start=2)  # mean 2/13: regret ~34.6 > 30
        self.assertEqual(eng.standing("vendor").selected, "drop")
        ev = [e.payload for e in events(eng, EventType.DECISION_EVALUATED) if e.payload["decision_id"] == "vendor"][-1]
        self.assertEqual(ev["trigger"], "voi_exceeded")
        self.assertEqual(ev["reversal_cost"]["total"], 30.0)
        self.assertEqual(eng.standing("integrate").selected, "wait")  # cascade still runs

    def test_without_costs_the_same_news_reverses_at_once(self):
        eng = build(direct=0.0, unwind_integrate=0.0, unwind_launch=0.0)
        bad_news(eng, 2)
        self.assertEqual(eng.standing("vendor").selected, "drop")


class Schedule(unittest.TestCase):
    def test_schedule_tracks_status(self):
        eng = build()
        bad_news(eng, 2)
        bad_news(eng, 8, start=2)
        sched = eng.auditor().amortization_schedule("vendor")
        statuses = [r["status"] for r in sched["schedule"]]
        self.assertEqual(statuses[0], "initial")
        self.assertIn("impaired", statuses)
        self.assertEqual(statuses[-1], "reversed")
        self.assertEqual(sched["reversals"], 1)
        self.assertIsNotNone(sched["first_impairment_seq"])
        self.assertGreater(sched["max_locked_in_regret"], 0)
        self.assertEqual(sched["currently"], "reversed")

    def test_override_appears(self):
        eng = build()
        eng.human_override("vendor", "drop", "cfo", "strategy", "consolidating")
        sched = eng.auditor().amortization_schedule("vendor")
        self.assertEqual((sched["schedule"][-1]["event"], sched["schedule"][-1]["status"]), ("override", "reversed"))


if __name__ == "__main__":
    unittest.main()
