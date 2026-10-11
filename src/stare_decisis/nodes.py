"""Node memory: individual pieces of evidence.

Nodes are never removed.  Retraction is a soft delete carrying a timestamp and
a coded reason; the node's value stays available so posteriors can be
corrected by subtraction or replay and so earlier decisions can be
reconstructed exactly as they were made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .vocab import NodeType, ReasoningForm, ReasonCode


@dataclass(frozen=True)
class LLMProvenance:
    """Identity of a generation.  The first output recorded for a key is the
    output of record; later calls with the same key return the frozen node
    instead of regenerating."""

    model: str
    prompt_hash: str
    temperature: float = 0.0
    seed: int | None = None

    def key(self) -> tuple[str, str, float, int | None]:
        return (self.model, self.prompt_hash, float(self.temperature), self.seed)


@dataclass(frozen=True)
class Retraction:
    tx_time: str
    reason_code: ReasonCode
    reason_text: str
    discovered_by: str
    discovery_ref: str | None
    seq: int


@dataclass
class Node:
    node_id: str
    node_type: NodeType
    value: Any
    value_hash: str
    reasoning_form: ReasoningForm
    source: dict[str, Any]
    valid_time: str | None
    tx_time: str
    seq: int
    llm_provenance: LLMProvenance | None = None
    retraction: Retraction | None = field(default=None)

    @property
    def live(self) -> bool:
        return self.retraction is None


class NodeMemory:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._frozen: dict[tuple, str] = {}

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __iter__(self):
        return iter(self._nodes.values())

    def get(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"unknown node: {node_id}") from None

    def frozen_for(self, provenance: LLMProvenance) -> Node | None:
        nid = self._frozen.get(provenance.key())
        return self._nodes[nid] if nid else None

    def add(self, node: Node) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"node already exists: {node.node_id}")
        self._nodes[node.node_id] = node
        if node.llm_provenance is not None:
            self._frozen[node.llm_provenance.key()] = node.node_id

    def mark_retracted(self, node_id: str, retraction: Retraction) -> None:
        node = self.get(node_id)
        if not node.live:
            raise ValueError(f"node already retracted: {node_id}")
        node.retraction = retraction
