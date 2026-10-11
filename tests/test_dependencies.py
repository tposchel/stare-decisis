import contextlib
import io
import json
import os
import tempfile
import unittest

from _support import IND, OBS, SRC, engine, events

from stare_decisis import (
    AgendaGroup,
    BetaBernoulli,
    Decision,
    EventType,
    Option,
    ReasonCode,
    UtilityFunction,
)
from stare_decisis.cli import main as cli


def scale(seg, option, upside):
    return lambda o, b: (100 * (b[seg]["mean"] - 0.5)) if o == upside else 0.0


def chain(path=":memory:"):
    """vendor -> integrate (assumes vendor kept) -> launch (evidence from integrate)."""
    eng = engine(path)
    eng.add_segment("fit", BetaBernoulli())
    eng.add_segment("capacity", BetaBernoulli())
    eng.add_segment("readiness", BetaBernoulli(), extract=lambda v: 1)
    eng.add_decision(Decision(
        "vendor", ("fit",), (Option("drop"), Option("keep")),
        UtilityFunction("vendor_u", 1, "proc", scale("fit", None, "keep")),
    ))
    eng.add_decision(Decision(
        "integrate", ("capacity",), (Option("wait"), Option("build", origin_nodes=("vendor-kept",))),
        UtilityFunction("integrate_u", 1, "eng", lambda o, b: 10.0 if o == "build" else 0.0),
    ))
    eng.add_decision(Decision(
        "launch", ("readiness",), (Option("hold"), Option("go")),
        UtilityFunction("launch_u", 1, "pm", scale("readiness", None, "go")),
        agenda_group=AgendaGroup.SCORE,  # earlier stage than its upstream: tests settling
    ))
    eng.assert_node("fit-1", OBS, 1, IND, SRC, segments=["fit"])
    eng.assert_node("cap-1", OBS, 1, IND, SRC, segments=["capacity"])
    eng.run()
    eng.publish_outcome("vendor", "vendor-kept")
    eng.run()
    eng.publish_outcome("integrate", "integrate-built", segments=["readiness"])
    eng.run()
    return eng


def reverse_vendor(eng):
    for i in range(4):
        eng.assert_node(f"fit-bad-{i}", OBS, 0, IND, SRC, segments=["fit"])
    return eng.run()


class Publishing(unittest.TestCase):
    def test_needs_a_standing_choice(self):
        eng = engine()
        eng.add_segment("fit", BetaBernoulli())
        eng.add_decision(Decision("v", ("fit",), (Option("a"),),
                                  UtilityFunction("u", 1, "x", lambda o, b: 0.0)))
        with self.assertRaises(ValueError):
            eng.publish_outcome("v", "v-out")

    def test_outcome_records_what_it_assumes(self):
        eng = chain()
        node = eng.nodes.get("vendor-kept")
        self.assertEqual(node.node_type.value, "decision_outcome")
        self.assertEqual(node.source["selected_option"], "keep")
        self.assertEqual(node.source["eval_id"], "vendor#1")
        self.assertEqual(eng.standing("launch").selected, "go")


class Cascade(unittest.TestCase):
    def test_reversal_cascades_and_settles_in_one_run(self):
        eng = chain()
        outcomes = reverse_vendor(eng)
        by = {(o.decision_id, o.kind, o.selected) for o in outcomes}
        self.assertIn(("vendor", "evaluated", "drop"), by)
        self.assertIn(("integrate", "evaluated", "wait"), by)
        self.assertIn(("launch", "unlinked", None), by)

        sup = [r.payload for r in events(eng, EventType.NODE_RETRACTED)]
        self.assertEqual([p["node_id"] for p in sup], ["vendor-kept", "integrate-built"])
        self.assertTrue(all(p["reason_code"] == "superseded" for p in sup))
        self.assertTrue(all(p["discovered_by"] == "engine:cascade" for p in sup))
        self.assertEqual(sup[0]["discovery_ref"], "vendor#2")
        self.assertEqual(eng.pending(), [])

    def test_unchanged_choice_does_not_cascade(self):
        eng = chain()
        eng.assert_node("fit-2", OBS, 1, IND, SRC, segments=["fit"])
        eng.run()
        self.assertEqual(events(eng, EventType.NODE_RETRACTED), [])

    def test_human_override_cascades(self):
        eng = chain()
        eng.human_override("vendor", "drop", "cfo", "strategy", "consolidating vendors")
        (r,) = events(eng, EventType.NODE_RETRACTED)
        self.assertEqual(r.payload["node_id"], "vendor-kept")
        self.assertTrue(r.payload["discovery_ref"].startswith("vendor#1/override@"))
        eng.run()
        self.assertEqual(eng.standing("integrate").selected, "wait")


class Exposure(unittest.TestCase):
    def test_transitive_exposure_and_curve(self):
        eng = chain()
        exp = eng.auditor().reversal_exposure("vendor")
        deps = {d["decision_id"]: d for d in exp["dependents"]}
        self.assertEqual(set(deps), {"integrate", "launch"})
        self.assertEqual((deps["integrate"]["depth"], deps["integrate"]["via_assumption"]), (1, ["vendor-kept"]))
        self.assertEqual((deps["launch"]["depth"], deps["launch"]["via_evidence"]), (2, ["integrate-built"]))
        self.assertEqual((exp["exposure"], exp["max_depth"]), (2.0, 2))
        self.assertEqual([p["dependent_count"] for p in exp["curve"]], [0, 1, 2])  # from first publication

        reverse_vendor(eng)
        after = eng.auditor().reversal_exposure("vendor")
        self.assertEqual(after["exposure"], 0.0)
        self.assertEqual([p["dependent_count"] for p in after["curve"]][-1], 0)
        self.assertFalse(after["published_nodes"][0]["live"])

    def test_as_of_and_weights(self):
        eng = chain()
        first = eng.auditor().reversal_exposure("vendor")["curve"][1]["seq"]
        early = eng.auditor().reversal_exposure("vendor", as_of_seq=first, weights={"integrate": 25.0})
        self.assertEqual([d["decision_id"] for d in early["dependents"]], ["integrate"])
        self.assertEqual(early["exposure"], 25.0)

    def test_leaf_has_no_exposure(self):
        self.assertEqual(chain().auditor().reversal_exposure("launch")["dependent_count"], 0)


class ReplayReasons(unittest.TestCase):
    def test_superseded_can_be_excluded_from_correction(self):
        eng = chain()
        reverse_vendor(eng)
        audit = eng.auditor()
        self.assertTrue(audit.replay_as_corrected("integrate", "integrate#2")["changed"])
        errors_only = [c.value for c in ReasonCode if c is not ReasonCode.SUPERSEDED]
        r = audit.replay_as_corrected("integrate", "integrate#2", reasons=errors_only)
        self.assertFalse(r["changed"])


class CLI(unittest.TestCase):
    def test_exposure_command(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "l.db")
            chain(path).log.close()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(cli(["exposure", path, "vendor"]), 0)
            self.assertEqual(json.loads(buf.getvalue())["dependent_count"], 2)


if __name__ == "__main__":
    unittest.main()
