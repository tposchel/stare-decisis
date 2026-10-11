import unittest

from _support import IND, OBS, SRC, events, observe, simple, utility, engine

from stare_decisis import (
    AgendaGroup,
    BetaBernoulli,
    ChoiceRule,
    Decision,
    EventType,
    LLMProvenance,
    NodeType,
    Option,
    ReasonCode,
    ReasoningForm,
    RecomputeMethod,
    Rule,
    RuleAction,
)

LLM = NodeType.LLM_HYPOTHESIS
ABD = ReasoningForm.ABDUCTION


def prov(prompt="p1"):
    return LLMProvenance("model-x", prompt, temperature=0.7, seed=1)


class LinkingAndScheduling(unittest.TestCase):
    def test_not_evaluated_until_linked(self):
        eng = simple()
        self.assertEqual(eng.run(), [])
        self.assertEqual(events(eng, EventType.DECISION_LINKED), [])
        observe(eng, "a", 1)
        self.assertEqual(len(events(eng, EventType.DECISION_LINKED)), 1)
        self.assertEqual(events(eng, EventType.SEGMENT_VERSION), [])  # staged, not computed
        out = eng.run()
        self.assertEqual([(o.kind, o.selected) for o in out], [("evaluated", "approve")])
        observe(eng, "b", 1)
        self.assertEqual(len(events(eng, EventType.DECISION_LINKED)), 1)

    def test_agenda_group_is_lazy(self):
        eng = simple()
        observe(eng, "a", 1)
        self.assertEqual(eng.run(AgendaGroup.SCORE), [])
        self.assertEqual(events(eng, EventType.SEGMENT_VERSION), [])
        self.assertEqual(len(eng.run(AgendaGroup.SELECT)), 1)

    def test_salience_order_and_shared_segment_flushed_once(self):
        eng = engine()
        eng.add_segment("rate", BetaBernoulli())
        shared = utility()
        for did, sal in (("low", 1), ("high", 5)):
            eng.add_decision(Decision(did, ("rate",), (Option("deny"), Option("approve")), shared, salience=sal))
        observe(eng, "a", 1)
        out = eng.run()
        self.assertEqual([o.decision_id for o in out], ["high", "low"])
        self.assertEqual(len(events(eng, EventType.SEGMENT_VERSION)), 1)
        evals = events(eng, EventType.DECISION_EVALUATED)
        self.assertEqual({e.payload["inputs"][0] for e in evals}, {"rate@1"})


class ValueOfInformation(unittest.TestCase):
    def test_unchanged_choice_is_skipped_and_logged(self):
        eng = simple()
        observe(eng, "a", 1)
        eng.run()
        observe(eng, "b", 1)
        (o,) = eng.run()
        self.assertEqual((o.kind, o.voi_estimate), ("skipped", 0.0))
        (skip,) = events(eng, EventType.REEVALUATION_SKIPPED)
        self.assertEqual(skip.payload["trigger_segments"], ["rate@2"])
        self.assertFalse(skip.payload["could_flip"])
        self.assertEqual(eng.run(), [])  # nothing new: no goal

    def test_threshold_tolerates_small_flips_but_not_large(self):
        eng = simple(threshold=20.0)
        observe(eng, "a", 1)  # mean 2/3
        eng.run()
        observe(eng, "b", 0)
        observe(eng, "c", 0)  # mean 2/5: deny better by 10
        (o,) = eng.run()
        self.assertEqual(o.kind, "skipped")
        self.assertAlmostEqual(o.voi_estimate, 10.0)
        skip = events(eng, EventType.REEVALUATION_SKIPPED)[-1].payload
        self.assertTrue(skip["could_flip"])
        self.assertEqual(skip["would_select"], "deny")
        observe(eng, "d", 0)
        observe(eng, "e", 0)  # mean 2/7: deny better by ~21.4
        (o,) = eng.run()
        self.assertEqual((o.kind, o.selected), ("evaluated", "deny"))
        last = events(eng, EventType.DECISION_EVALUATED)[-1].payload
        self.assertEqual(last["trigger"], "voi_exceeded")
        self.assertEqual(last["supersedes"], "d1#1")


class Registries(unittest.TestCase):
    def test_utility_refs_are_immutable(self):
        eng = engine()
        eng.add_segment("rate", BetaBernoulli())
        eng.add_decision(Decision("a", ("rate",), (Option("x"),), utility()))
        with self.assertRaises(ValueError):
            eng.add_decision(Decision("b", ("rate",), (Option("x"),), utility()))


class Policy(unittest.TestCase):
    def build(self, rule):
        eng = engine()
        eng.add_segment("rate", BetaBernoulli())
        eng.add_rule(rule)
        eng.add_decision(Decision("d1", ("rate",), (Option("deny"), Option("approve")), utility(),
                                  constraints=(rule.rule_id,)))
        return eng

    def test_block_rule(self):
        eng = self.build(Rule("strict", 1, "policy", lambda c: c["option_id"] == "approve"
                              and c["beliefs"]["rate"]["mean"] < 0.9))
        observe(eng, "a", 1)
        (o,) = eng.run()
        self.assertEqual(o.selected, "deny")
        (fired,) = events(eng, EventType.RULE_FIRED)
        self.assertEqual((fired.payload["rule"], fired.payload["option_id"]), ("strict@1", "approve"))
        ev = events(eng, EventType.DECISION_EVALUATED)[0].payload
        self.assertIn({"rule": "strict@1", "author": "policy", "option_id": "approve",
                       "matched_nodes": [], "result": "fail"}, ev["constraints"])

    def test_mandate_rule(self):
        eng = self.build(Rule("freeze", 1, "policy", lambda c: c["beliefs"]["rate"]["n"] >= 2,
                              action=RuleAction.MANDATE, target_option="deny"))
        observe(eng, "a", 1)
        self.assertEqual(eng.run()[0].selected, "approve")
        observe(eng, "b", 1)
        (o,) = eng.run()
        self.assertEqual((o.kind, o.selected), ("evaluated", "deny"))
        self.assertEqual(eng.standing("d1").choice_rule, ChoiceRule.POLICY_MANDATE)

    def test_rules_are_immutable(self):
        eng = engine()
        r = Rule("x", 1, "a", lambda c: False)
        eng.add_rule(r)
        with self.assertRaises(ValueError):
            eng.add_rule(Rule("x", 1, "a", lambda c: True))
        with self.assertRaises(ValueError):
            eng.add_rule(Rule("y", 1, "", lambda c: True))


class LLMFreezing(unittest.TestCase):
    def test_hypothesis_requires_provenance(self):
        eng = simple()
        with self.assertRaises(ValueError):
            eng.assert_node("h", LLM, "x", ABD, SRC)

    def test_first_output_is_output_of_record(self):
        eng = simple()
        first = eng.assert_node("h1", LLM, {"claim": "A"}, ABD, SRC, llm_provenance=prov())
        n = len(eng.log)
        again = eng.assert_node("h2", LLM, {"claim": "B"}, ABD, SRC, llm_provenance=prov())
        self.assertIs(again, first)
        self.assertEqual(len(eng.log), n)
        rec = eng.log.get(first.seq).payload
        self.assertEqual(rec["llm_provenance"]["output_hash"], first.value_hash)


class Retraction(unittest.TestCase):
    def test_soft_delete_and_subtraction(self):
        eng = simple()
        observe(eng, "a", 1)
        observe(eng, "b", 1)
        eng.run()
        rec = eng.retract("b", ReasonCode.SOURCE_ERROR, "duplicate feed row", "recon-job")
        self.assertEqual(rec.payload["affected_segments"], ["rate@1"])
        self.assertFalse(eng.nodes.get("b").live)
        self.assertEqual(eng.nodes.get("b").value, 1)  # value kept
        eng.run()
        v = events(eng, EventType.SEGMENT_VERSION)[-1].payload
        self.assertEqual((v["version"], v["recompute_method"], v["retracted_nodes"]),
                         (2, RecomputeMethod.SUBTRACTION.value, ["b"]))
        self.assertEqual(v["consumed_nodes"], ["a"])
        with self.assertRaises(ValueError):
            eng.retract("b", ReasonCode.SOURCE_ERROR, "again", "x")

    def test_retraction_before_flush_is_pending(self):
        eng = simple()
        observe(eng, "a", 1)
        rec = eng.retract("a", ReasonCode.EXTRACTION_ERROR, "bad parse", "x")
        self.assertEqual((rec.payload["affected_segments"], rec.payload["pending_segments"]), ([], ["rate"]))
        self.assertEqual(eng.run(), [])

    def test_drain_unlinks_and_relinks(self):
        eng = simple()
        observe(eng, "a", 1)
        eng.run()
        eng.retract("a", ReasonCode.SOURCE_ERROR, "wrong account", "x")
        (o,) = eng.run()
        self.assertEqual(o.kind, "unlinked")
        (u,) = events(eng, EventType.DECISION_UNLINKED)
        self.assertEqual(u.payload["standing_option"], "approve")
        observe(eng, "c", 0)
        self.assertEqual(len(events(eng, EventType.DECISION_LINKED)), 2)
        self.assertEqual(eng.run()[0].selected, "deny")

    def test_retracted_origin_withdraws_option(self):
        eng = simple(origin=("h1",))
        eng.assert_node("h1", LLM, {"claim": "x"}, ABD, SRC, llm_provenance=prov())
        observe(eng, "a", 1)
        self.assertEqual(eng.run()[0].selected, "approve")
        eng.retract("h1", ReasonCode.MODEL_HALLUCINATION, "fabricated", "analyst")
        (o,) = eng.run()
        self.assertEqual((o.kind, o.selected), ("evaluated", "deny"))
        ev = events(eng, EventType.DECISION_EVALUATED)[-1].payload
        self.assertEqual((ev["trigger"], ev["withdrawn_options"]), ("option_origin_retracted", ["approve"]))


class Humans(unittest.TestCase):
    def test_override_stands_with_system_recommendation(self):
        eng = simple()
        observe(eng, "a", 1)
        eng.run()
        eng.human_override("d1", "deny", "officer:kim", "customer_request", "applicant withdrew")
        observe(eng, "b", 1)
        (o,) = eng.run()
        self.assertEqual((o.kind, o.selected), ("evaluated", "deny"))
        ev = events(eng, EventType.DECISION_EVALUATED)[-1].payload
        self.assertEqual((ev["choice_rule"], ev["system_recommendation"], ev["actor"]),
                         ("human_override", "approve", "officer:kim"))
        self.assertIn("human_override_stands", ev["flags"])

    def test_policy_revokes_override(self):
        eng = engine()
        eng.add_segment("rate", BetaBernoulli())
        eng.add_rule(Rule("no_deny_if_strong", 1, "policy",
                          lambda c: c["option_id"] == "deny" and c["beliefs"]["rate"]["mean"] > 0.7))
        eng.add_decision(Decision("d1", ("rate",), (Option("deny"), Option("approve")), utility(),
                                  constraints=("no_deny_if_strong",)))
        observe(eng, "a", 1)
        eng.run()
        eng.human_override("d1", "deny", "officer:kim", "judgment", "gut call")
        observe(eng, "b", 1)
        observe(eng, "c", 1)
        (o,) = eng.run()
        self.assertEqual(o.selected, "approve")
        ev = events(eng, EventType.DECISION_EVALUATED)[-1].payload
        self.assertIn("human_override_revoked_by_policy", ev["flags"])

    def test_utility_change_forces_evaluation(self):
        eng = simple()
        observe(eng, "a", 1)
        eng.run()
        eng.set_utility("d1", utility(version=2, bar=0.9))
        (o,) = eng.run()
        self.assertEqual((o.kind, o.selected), ("evaluated", "deny"))
        ev = events(eng, EventType.DECISION_EVALUATED)[-1].payload
        self.assertEqual((ev["trigger"], ev["utility_fn_ref"]), ("utility_changed", "approve_value@2"))


if __name__ == "__main__":
    unittest.main()
