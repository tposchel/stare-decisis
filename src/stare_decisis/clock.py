"""Transaction-time clocks.

`tx_time` is when the ledger recorded something.  `valid_time` is supplied by
the caller and says when the fact held in the world.  The two are never
conflated.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol


def parse_instant(s: str) -> datetime:
    """Parse an ISO date or datetime.  Dates and naive datetimes are taken as UTC."""
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


class Clock(Protocol):
    def now(self) -> str: ...


class SystemClock:
    def now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class ManualClock:
    """Deterministic clock for tests and replays: advances by `step_seconds` per read."""

    def __init__(self, start: str = "2026-01-01T00:00:00+00:00", step_seconds: float = 1.0):
        self._t = datetime.fromisoformat(start)
        if self._t.tzinfo is None:
            raise ValueError("ManualClock start must be timezone-aware")
        self._step = timedelta(seconds=step_seconds)

    def now(self) -> str:
        t = self._t
        self._t += self._step
        return t.astimezone(timezone.utc).isoformat(timespec="microseconds")
