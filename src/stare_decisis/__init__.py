"""stare-decisis: auditable decisions across rules, probabilistic models and LLM output."""

from .audit import AuditError, Auditor
from .clock import ManualClock, SystemClock
from .decisions import Decision, Option, Outcome, UtilityFunction
from .engine import Engine
from .log import AuditLog, ChainCheck, Record
from .models import BetaBernoulli, GaussianMixtureEM, NormalKnownVariance, SegmentModel
from .nodes import LLMProvenance, Node
from .rules import Rule
from .vocab import (
    AgendaGroup,
    ChoiceRule,
    EventType,
    NodeType,
    ReasonCode,
    ReasoningForm,
    RecomputeMethod,
    RuleAction,
    Trigger,
)

__version__ = "0.1.0"

__all__ = [
    "AgendaGroup",
    "AuditError",
    "AuditLog",
    "Auditor",
    "BetaBernoulli",
    "ChainCheck",
    "ChoiceRule",
    "Decision",
    "Engine",
    "EventType",
    "GaussianMixtureEM",
    "LLMProvenance",
    "ManualClock",
    "Node",
    "NodeType",
    "NormalKnownVariance",
    "Option",
    "Outcome",
    "ReasonCode",
    "ReasoningForm",
    "RecomputeMethod",
    "Record",
    "Rule",
    "RuleAction",
    "SegmentModel",
    "SystemClock",
    "Trigger",
    "UtilityFunction",
]
