import contextlib
import io
import json
import os
import tempfile
import unittest

from _support import SRC, observe, simple, utility

from stare_decisis import AuditError, Auditor, LLMProvenance, NodeType, ReasonCode, ReasoningForm
from stare_decisis.cli import main as cli

LLM, ABD = NodeType.LLM_HYPOTHESIS, ReasoningForm.ABDUCTION


def scenario(path=":memory:"):
    eng = simple(path, origin=("h1",))
    eng.assert_node("h1", LLM, {"claim": "income rose"}, ABD, {"id": "llm/assist"},
                    llm_provenance=LLMProvenance("m", "prompt-A", 0.2, 1))
    for nid, v in (("a", 1), ("b", 1), ("c", 1), ("d", 0)):
        observe(eng, nid, v)
    eng.run()  # d1#1: approve (mean 4/6)
    return eng


class AsMade(unittest.TestCase):
    def test_excludes_later_records(self):
        eng = scenario()
        observe(eng, "late", 0)
        eng.retract("a", ReasonCode.SOURCE_ERROR, "dup", "recon")
        eng.run()
        made = eng.auditor().reconstruct_as_made("d1", "d1#1")
        self.assertEqual(list(made["segments"]), ["rate@1"])
        self.assertNotIn("late", made["nodes"])
        self.assertEqual(made["retractions_known_at_time"], {})
        self.assertEqual(made["evaluation"]["selected_option"], "approve")

    def test_unknown_evaluation(self):
        with self.assertRaises(AuditError):
            scenario().auditor().reconstruct_as_made("d1", "d1#9")


class AsCorrected(unittest.TestCase):
    def test_retractions_change_the_answer(self):
        eng = scenario()
        for nid in ("a", "b", "c"):
            eng.retract(nid, ReasonCode.SOURCE_ERROR, "test account", "recon")
        r = eng.auditor().replay_as_corrected("d1", "d1#1")  # only d=0 left: mean 1/3
        self.assertTrue(r["changed"])
        self.assertEqual((r["as_made"]["selected"], r["as_corrected"]["selected"]), ("approve", "deny"))
        self.assertEqual(r["excluded_nodes"], {"rate@1": ["a", "b", "c"]})

    def test_as_of_seq_limits_corrections(self):
        eng = scenario()
        rec = eng.retract("a", ReasonCode.SOURCE_ERROR, "x", "recon")
        eng.retract("b", ReasonCode.SOURCE_ERROR, "x", "recon")
        r = eng.auditor().replay_as_corrected("d1", "d1#1", as_of_seq=rec.seq)
        self.assertEqual(r["excluded_nodes"], {"rate@1": ["a"]})
        self.assertFalse(r["changed"])  # mean 3/5 still approves

    def test_uses_recorded_utility_version(self):
        eng = scenario()
        eng.set_utility("d1", utility(version=2, bar=0.99))
        eng.run()
        r = eng.auditor().replay_as_corrected("d1", "d1#1")
        self.assertEqual(r["as_corrected"]["selected"], "approve")
        self.assertAlmostEqual(r["as_corrected"]["expected_utility"]["approve"],
                               r["as_made"]["expected_utility"]["approve"])

    def test_retracted_origin_withdraws_option(self):
        eng = scenario()
        eng.retract("h1", ReasonCode.MODEL_HALLUCINATION, "fabricated", "analyst")
        r = eng.auditor().replay_as_corrected("d1", "d1#1")
        self.assertEqual((r["withdrawn_options"], r["as_corrected"]["selected"]), (["approve"], "deny"))

    def test_needs_engine(self):
        eng = scenario()
        with self.assertRaises(AuditError):
            Auditor(eng.log).replay_as_corrected("d1", "d1#1")


class Queries(unittest.TestCase):
    def test_impact(self):
        eng = scenario()
        observe(eng, "e", 1)
        eng.run()  # skipped, no new evaluation
        imp = eng.auditor().retraction_impact("a")
        self.assertEqual(imp["segment_versions"], ["rate@1", "rate@2"])
        self.assertEqual([e["eval_id"] for e in imp["evaluations"]], ["d1#1"])
        origin = eng.auditor().retraction_impact("h1")["evaluations"][0]
        self.assertTrue(origin["selected_depends_on_origin"])

    def test_clusters(self):
        eng = scenario()
        eng.assert_node("h2", LLM, {"claim": "y"}, ABD, SRC, llm_provenance=LLMProvenance("m", "prompt-A", 0.2, 2))
        eng.assert_node("h3", LLM, {"claim": "z"}, ABD, SRC, llm_provenance=LLMProvenance("m", "prompt-B", 0.2, 1))
        for n in ("h1", "h2", "h3"):
            eng.retract(n, ReasonCode.MODEL_HALLUCINATION, "fabricated", "analyst")
        clusters = eng.auditor().retraction_clusters()
        self.assertEqual([(c["key"], c["count"]) for c in clusters], [("prompt-A", 2), ("prompt-B", 1)])

    def test_trail_flags_llm_abduction(self):
        trail = scenario().auditor().reasoning_trail("d1", "d1#1")
        self.assertIn("selected_option_rests_on_unverified_llm_abduction", trail["flags"])
        self.assertEqual(trail["form_counts"], {"abduction": 1, "induction": 4})

    def test_skipped_query(self):
        eng = scenario()
        observe(eng, "e", 1)
        eng.run()
        (s,) = eng.auditor().skipped_reevaluations("d1")
        self.assertEqual(s["standing_eval_id"], "d1#1")


class CLI(unittest.TestCase):
    def run_cli(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli(list(args))
        return code, buf.getvalue()

    def test_verify_and_trail(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "l.db")
            scenario(path).log.close()
            code, out = self.run_cli("verify", path)
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out)["ok"])
            code, out = self.run_cli("trail", path, "d1", "d1#1")
            self.assertEqual(json.loads(out)["selected_option"], "approve")
            code, out = self.run_cli("history", path, "d1")
            self.assertGreater(len(json.loads(out)), 1)


if __name__ == "__main__":
    unittest.main()
