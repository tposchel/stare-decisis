"""Segment memory: beliefs shared by every decision whose path includes them.

Borrowed from PHREAK:

- staging: inserts and deletes queue here and cost nothing until the segment
  is flushed, which happens only when a decision on its path is evaluated
- linking: a segment is linked once it has (or has staged) at least one live
  node; a decision is linked only when every segment on its path is
- set-oriented flush: all staged changes are applied in one pass, producing
  exactly one new version

Every flush yields an immutable, numbered `SegmentVersion` listing the nodes it
consumed.  Those lists are the dependency edges used by retraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from .hashing import content_hash
from .models import SegmentModel
from .vocab import RecomputeMethod, Trigger


@dataclass(frozen=True)
class SegmentVersion:
    segment_id: str
    version: int
    consumed_nodes: tuple[str, ...]
    state: dict[str, Any]
    summary: dict[str, Any]
    sufficient_stats_hash: str
    model_spec_hash: str
    prior_ref: str | None
    recompute_method: RecomputeMethod
    trigger: Trigger
    inserted_nodes: tuple[str, ...]
    retracted_nodes: tuple[str, ...]
    expired_nodes: tuple[str, ...] = ()

    @property
    def ref(self) -> str:
        return f"{self.segment_id}@{self.version}"


class Segment:
    def __init__(
        self,
        segment_id: str,
        model: SegmentModel,
        prior_ref: str | None = None,
        extract: Callable[[Any], Any] | None = None,
        freshness: timedelta | None = None,
    ):
        if freshness is not None and freshness <= timedelta(0):
            raise ValueError("freshness window must be positive")
        self.segment_id = segment_id
        self.freshness = freshness
        self.model = model
        self.prior_ref = prior_ref
        self.extract: Callable[[Any], Any] = extract or (lambda v: v)
        self.version = 0
        self.state: dict[str, Any] = model.empty_state()
        self.consumed: list[str] = []
        self.versions: dict[int, SegmentVersion] = {}
        self._staged: list[tuple[str, str]] = []

    # -- staging (cheap, no belief work) ------------------------------------

    def stage_insert(self, node_id: str) -> None:
        self._staged.append(("insert", node_id))

    def stage_delete(self, node_id: str) -> None:
        self._staged.append(("delete", node_id))

    def stage_expire(self, node_id: str) -> None:
        """Remove evidence that aged out of the freshness window.  Not an error,
        so it is tracked apart from retraction."""
        self._staged.append(("expire", node_id))

    @property
    def dirty(self) -> bool:
        return bool(self._staged)

    def projected_members(self) -> list[str]:
        members = list(self.consumed)
        for op, nid in self._staged:
            if op == "insert" and nid not in members:
                members.append(nid)
            elif op in ("delete", "expire") and nid in members:
                members.remove(nid)
        return members

    @property
    def linked(self) -> bool:
        return bool(self.projected_members())

    @property
    def summary(self) -> dict[str, Any]:
        return self.model.summary(self.state)

    def contains(self, node_id: str) -> bool:
        return node_id in self.consumed or any(nid == node_id for _, nid in self._staged)

    # -- flush (set-oriented, one version per flush) ------------------------

    def flush(self, value_of: Callable[[str], Any]) -> SegmentVersion | None:
        if not self._staged:
            return None
        staged, self._staged = self._staged, []

        consumed = list(self.consumed)
        inserts: list[str] = []
        deletes: list[str] = []
        expired: list[str] = []
        for op, nid in staged:
            if op == "insert":
                if nid in consumed:
                    raise ValueError(f"{self.segment_id}: node {nid} inserted twice")
                consumed.append(nid)
                inserts.append(nid)
            else:
                if nid not in consumed:
                    continue  # already gone; deleting twice is a no-op
                consumed.remove(nid)
                if nid in inserts:
                    inserts.remove(nid)  # inserted and removed within one batch
                else:
                    (deletes if op == "delete" else expired).append(nid)

        removed = deletes + expired
        if not inserts and not removed:
            return None

        m = self.model
        if removed:
            method = RecomputeMethod.SUBTRACTION if (m.subtractable and m.incremental) else RecomputeMethod.REPLAY
        else:
            method = RecomputeMethod.INCREMENTAL if m.incremental else RecomputeMethod.REPLAY

        if method is RecomputeMethod.REPLAY:
            state = m.fit([self.extract(value_of(n)) for n in consumed])
        else:
            state = dict(self.state)
            for nid in removed:
                state = m.subtract(state, self.extract(value_of(nid)))
            for nid in inserts:
                state = m.add(state, self.extract(value_of(nid)))

        self.version += 1
        self.state = state
        self.consumed = consumed
        v = SegmentVersion(
            segment_id=self.segment_id,
            version=self.version,
            consumed_nodes=tuple(consumed),
            state=dict(state),
            summary=m.summary(state),
            sufficient_stats_hash=content_hash(state),
            model_spec_hash=m.spec_hash(),
            prior_ref=self.prior_ref,
            recompute_method=method,
            trigger=(
                Trigger.RETRACTION if deletes else Trigger.SCHEDULED if expired else Trigger.ASSERTION
            ),
            inserted_nodes=tuple(inserts),
            retracted_nodes=tuple(deletes),
            expired_nodes=tuple(expired),
        )
        self.versions[self.version] = v
        return v
