import unittest
from datetime import timedelta

from _support import IND, OBS, SRC, engine, events, utility

from stare_decisis import BetaBernoulli, Decision, EventType, Option, RecomputeMethod, Trigger


def build(window_days=30):
    eng = engine()
    eng.add_segment("rate", BetaBernoulli(), freshness=timedelta(days=window_days))
    eng.add_decision(Decision("d1", ("rate",), (Option("deny"), Option("approve")), utility()))
    return eng


def obs(eng, nid, v, when):
    return eng.assert_node(nid, OBS, v, IND, SRC, valid_time=when, segments=["rate"])


class Expiry(unittest.TestCase):
    def test_stale_evidence_expires_on_schedule(self):
        eng = build()
        obs(eng, "old0", 0, "2026-01-01")
        obs(eng, "old1", 0, "2026-01-02")
        obs(eng, "new", 1, "2026-02-20")
        self.assertEqual(eng.run()[0].selected, "deny")  # mean 2/5

        recs = eng.expire("2026-03-01")
        self.assertEqual([r.payload["node_id"] for r in recs], ["old0", "old1"])
        self.assertEqual(recs[0].payload["affected_segments"], ["rate@1"])
        self.assertEqual(recs[0].payload["aged_from"], "valid_time")

        (o,) = eng.run()
        self.assertEqual((o.kind, o.selected), ("evaluated", "approve"))  # mean 2/3
        v = events(eng, EventType.SEGMENT_VERSION)[-1].payload
        self.assertEqual(v["trigger"], Trigger.SCHEDULED.value)
        self.assertEqual(v["recompute_method"], RecomputeMethod.SUBTRACTION.value)
        self.assertEqual((v["expired_nodes"], v["retracted_nodes"]), (["old0", "old1"], []))

    def test_expiry_is_idempotent_and_not_a_retraction(self):
        eng = build()
        obs(eng, "old", 1, "2026-01-01")
        eng.run()
        eng.expire("2026-06-01")
        self.assertEqual(eng.expire("2026-06-01"), [])
        self.assertTrue(eng.nodes.get("old").live)
        self.assertEqual(events(eng, EventType.NODE_RETRACTED), [])

    def test_full_expiry_unlinks(self):
        eng = build()
        obs(eng, "old", 1, "2026-01-01")
        eng.run()
        eng.expire("2026-06-01")
        self.assertEqual(eng.run()[0].kind, "unlinked")

    def test_as_corrected_does_not_treat_expiry_as_error(self):
        eng = build()
        obs(eng, "a", 1, "2026-01-01")
        obs(eng, "b", 1, "2026-01-02")
        eng.run()
        eng.expire("2026-06-01")
        eng.run()
        r = eng.auditor().replay_as_corrected("d1", "d1#1")
        self.assertFalse(r["changed"])
        self.assertEqual(r["excluded_nodes"], {})

    def test_segments_without_window_never_expire(self):
        eng = engine()
        eng.add_segment("rate", BetaBernoulli())
        eng.assert_node("x", OBS, 1, IND, SRC, valid_time="1999-01-01", segments=["rate"])
        self.assertEqual(eng.expire("2026-06-01"), [])

    def test_window_must_be_positive(self):
        with self.assertRaises(ValueError):
            engine().add_segment("s", BetaBernoulli(), freshness=timedelta(0))


if __name__ == "__main__":
    unittest.main()
