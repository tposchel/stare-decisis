"""Controlled vocabularies.  Values are what appear in the log."""

from __future__ import annotations

from enum import Enum


class _Vocab(str, Enum):
    def __str__(self) -> str:
        return self.value


class NodeType(_Vocab):
    OBSERVATION = "observation"
    LLM_HYPOTHESIS = "llm_hypothesis"
    LATENT_ESTIMATE = "latent_estimate"
    RULE_FACT = "rule_fact"
    DECISION_OUTCOME = "decision_outcome"


class ReasoningForm(_Vocab):
    DEDUCTION = "deduction"
    INDUCTION = "induction"
    ABDUCTION = "abduction"
    ANALOGY = "analogy"
    AUTHORITY = "authority"


class ReasonCode(_Vocab):
    SOURCE_ERROR = "source_error"
    EXTRACTION_ERROR = "extraction_error"
    MODEL_HALLUCINATION = "model_hallucination"
    SUPERSEDED = "superseded"
    ADVERSARIAL = "adversarial"


class AgendaGroup(_Vocab):
    PROPOSE = "propose"
    SCORE = "score"
    SELECT = "select"
    ENFORCE = "enforce"


AGENDA_ORDER: tuple[AgendaGroup, ...] = (
    AgendaGroup.PROPOSE,
    AgendaGroup.SCORE,
    AgendaGroup.SELECT,
    AgendaGroup.ENFORCE,
)


class ChoiceRule(_Vocab):
    MAX_EU = "max_eu"
    POLICY_MANDATE = "policy_mandate"
    HUMAN_OVERRIDE = "human_override"


class RecomputeMethod(_Vocab):
    INCREMENTAL = "incremental"
    SUBTRACTION = "subtraction"
    REPLAY = "replay"


class Trigger(_Vocab):
    ASSERTION = "assertion"
    RETRACTION = "retraction"
    SCHEDULED = "scheduled"


class RuleAction(_Vocab):
    BLOCK = "block"
    MANDATE = "mandate"


class EventType(_Vocab):
    NODE_ASSERTED = "node_asserted"
    NODE_RETRACTED = "node_retracted"
    NODE_EXPIRED = "node_expired"
    SEGMENT_VERSION = "segment_version"
    DECISION_LINKED = "decision_linked"
    DECISION_UNLINKED = "decision_unlinked"
    DECISION_EVALUATED = "decision_evaluated"
    REEVALUATION_SKIPPED = "reevaluation_skipped"
    RULE_FIRED = "rule_fired"
    HUMAN_OVERRIDE = "human_override"
