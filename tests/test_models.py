import unittest

import _support  # noqa: F401  (path setup)

from stare_decisis import BetaBernoulli, GaussianMixtureEM, NormalKnownVariance, RecomputeMethod, Trigger
from stare_decisis.segments import Segment


class BetaTests(unittest.TestCase):
    def test_subtraction_matches_refit(self):
        m = BetaBernoulli(2, 3)
        xs = [1, 0, 1, 1, 0, 1]
        s = m.fit(xs)
        s = m.subtract(s, 0)
        self.assertEqual(m.summary(s), m.summary(m.fit([1, 1, 1, 0, 1])))

    def test_cannot_subtract_absent(self):
        m = BetaBernoulli()
        with self.assertRaises(ValueError):
            m.subtract(m.fit([1]), 0)

    def test_rejects_non_binary(self):
        with self.assertRaises(ValueError):
            BetaBernoulli().add(BetaBernoulli().empty_state(), 0.3)


class NormalTests(unittest.TestCase):
    def test_known_posterior(self):
        m = NormalKnownVariance(0.0, 1.0, 1.0)
        s = m.summary(m.fit([2.0]))
        self.assertAlmostEqual(s["mean"], 1.0)
        self.assertAlmostEqual(s["var"], 0.5)

    def test_subtraction_matches_refit(self):
        m = NormalKnownVariance(1.0, 4.0, 0.5)
        s = m.subtract(m.fit([0.3, 1.7, 2.2]), 1.7)
        ref = m.summary(m.fit([0.3, 2.2]))
        for k in ("mean", "var"):
            self.assertAlmostEqual(m.summary(s)[k], ref[k])


class EMTests(unittest.TestCase):
    def test_recovers_two_regimes(self):
        xs = [10, 11, 9, 10.5, 9.5, 50, 51, 49, 50.5]
        s = GaussianMixtureEM(k=2).fit(xs)
        self.assertAlmostEqual(s["means"][0], 10.0, places=3)
        self.assertAlmostEqual(s["means"][1], 50.125, places=3)
        self.assertAlmostEqual(s["weights"][1], 4 / 9, places=3)
        self.assertTrue(s["converged"])

    def test_deterministic(self):
        xs = [1, 2, 3, 20, 21, 22, 5]
        m = GaussianMixtureEM()
        self.assertEqual(m.fit(xs), m.fit(xs))

    def test_degenerate_inputs(self):
        m = GaussianMixtureEM()
        self.assertEqual(m.summary(m.fit([]))["n"], 0)
        s = m.fit([4, 4, 4])
        self.assertEqual(s["weights"], [1.0])

    def test_not_incremental(self):
        m = GaussianMixtureEM()
        self.assertFalse(m.incremental or m.subtractable)


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.values = {"a": 1, "b": 0, "c": 1}

    def flush(self, seg):
        return seg.flush(self.values.__getitem__)

    def test_staging_defers_work_and_sets_link_bit(self):
        seg = Segment("s", BetaBernoulli())
        self.assertFalse(seg.linked)
        seg.stage_insert("a")
        self.assertTrue(seg.linked)
        self.assertEqual(seg.version, 0)
        self.assertEqual(seg.state, {"successes": 0, "failures": 0})
        v = self.flush(seg)
        self.assertEqual((v.version, v.recompute_method, v.trigger), (1, RecomputeMethod.INCREMENTAL, Trigger.ASSERTION))

    def test_one_version_per_flush(self):
        seg = Segment("s", BetaBernoulli())
        for n in "abc":
            seg.stage_insert(n)
        v = self.flush(seg)
        self.assertEqual(v.version, 1)
        self.assertEqual(v.consumed_nodes, ("a", "b", "c"))
        self.assertIsNone(self.flush(seg))

    def test_delete_uses_subtraction_when_possible(self):
        seg = Segment("s", BetaBernoulli())
        seg.stage_insert("a")
        seg.stage_insert("b")
        self.flush(seg)
        seg.stage_delete("b")
        v = self.flush(seg)
        self.assertEqual((v.recompute_method, v.trigger), (RecomputeMethod.SUBTRACTION, Trigger.RETRACTION))
        self.assertEqual(v.retracted_nodes, ("b",))
        self.assertEqual(v.state, {"successes": 1, "failures": 0})

    def test_em_always_replays(self):
        seg = Segment("s", GaussianMixtureEM())
        seg.stage_insert("a")
        self.assertEqual(self.flush(seg).recompute_method, RecomputeMethod.REPLAY)

    def test_insert_then_delete_in_one_batch_nets_out(self):
        seg = Segment("s", BetaBernoulli())
        seg.stage_insert("a")
        seg.stage_delete("a")
        self.assertFalse(seg.linked)
        self.assertIsNone(self.flush(seg))
        self.assertEqual(seg.version, 0)

    def test_double_insert_rejected(self):
        seg = Segment("s", BetaBernoulli())
        seg.stage_insert("a")
        seg.stage_insert("a")
        with self.assertRaises(ValueError):
            self.flush(seg)


if __name__ == "__main__":
    unittest.main()
