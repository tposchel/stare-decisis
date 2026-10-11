"""Append-only, hash-chained, bitemporal audit log on SQLite.

Each record's hash covers its sequence number, type, both timestamps, payload
and the previous record's hash.  Altering any record breaks verification of
every record after it.  SQLite triggers reject UPDATE and DELETE so the
append-only property holds against ordinary writes; `verify()` catches the
rest.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterator

from .clock import Clock, SystemClock
from .hashing import GENESIS_HASH, canonical, normalize, sha256_hex
from .vocab import EventType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY,
    event_type  TEXT NOT NULL,
    tx_time     TEXT NOT NULL,
    valid_time  TEXT,
    payload     TEXT NOT NULL,
    prev_hash   TEXT NOT NULL,
    hash        TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS events_by_type ON events(event_type);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""

_EVENT_TYPES = frozenset(e.value for e in EventType)


@dataclass(frozen=True)
class Record:
    seq: int
    event_type: str
    tx_time: str
    valid_time: str | None
    payload: dict[str, Any]
    prev_hash: str
    hash: str


@dataclass(frozen=True)
class ChainCheck:
    ok: bool
    checked: int
    first_bad_seq: int | None = None
    reason: str | None = None


def record_hash(
    seq: int,
    event_type: str,
    tx_time: str,
    valid_time: str | None,
    payload: dict[str, Any],
    prev_hash: str,
) -> str:
    return sha256_hex(
        canonical(
            {
                "seq": seq,
                "event_type": event_type,
                "tx_time": tx_time,
                "valid_time": valid_time,
                "payload": payload,
                "prev_hash": prev_hash,
            }
        )
    )


class AuditLog:
    def __init__(self, path: str = ":memory:", clock: Clock | None = None):
        self.path = path
        self.clock: Clock = clock if clock is not None else SystemClock()
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- writing -----------------------------------------------------------

    def append(
        self,
        event_type: EventType | str,
        payload: dict[str, Any],
        valid_time: str | None = None,
    ) -> Record:
        etype = str(event_type)
        if etype not in _EVENT_TYPES:
            raise ValueError(f"unknown event type: {etype}")
        body = normalize(payload)
        with self._conn:
            row = self._conn.execute(
                "SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            seq = (row[0] + 1) if row else 1
            prev = row[1] if row else GENESIS_HASH
            tx = self.clock.now()
            h = record_hash(seq, etype, tx, valid_time, body, prev)
            self._conn.execute(
                "INSERT INTO events (seq, event_type, tx_time, valid_time, payload, prev_hash, hash)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (seq, etype, tx, valid_time, canonical(body), prev, h),
            )
        return Record(seq, etype, tx, valid_time, body, prev, h)

    # -- reading -----------------------------------------------------------

    def _rows(self, sql: str, params: tuple = ()) -> Iterator[Record]:
        for seq, etype, tx, vt, payload, prev, h in self._conn.execute(sql, params):
            yield Record(seq, etype, tx, vt, json.loads(payload), prev, h)

    def records(
        self,
        event_type: EventType | str | None = None,
        *,
        until_seq: int | None = None,
    ) -> list[Record]:
        clauses, params = [], []
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(str(event_type))
        if until_seq is not None:
            clauses.append("seq <= ?")
            params.append(until_seq)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return list(
            self._rows(
                "SELECT seq, event_type, tx_time, valid_time, payload, prev_hash, hash"
                f" FROM events{where} ORDER BY seq",
                tuple(params),
            )
        )

    def get(self, seq: int) -> Record:
        rows = list(
            self._rows(
                "SELECT seq, event_type, tx_time, valid_time, payload, prev_hash, hash"
                " FROM events WHERE seq = ?",
                (seq,),
            )
        )
        if not rows:
            raise KeyError(seq)
        return rows[0]

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    # -- integrity ---------------------------------------------------------

    def verify(self) -> ChainCheck:
        prev = GENESIS_HASH
        expected_seq = 1
        checked = 0
        for seq, etype, tx, vt, payload_json, prev_hash, h in self._conn.execute(
            "SELECT seq, event_type, tx_time, valid_time, payload, prev_hash, hash"
            " FROM events ORDER BY seq"
        ):
            if seq != expected_seq:
                return ChainCheck(False, checked, seq, f"sequence gap: expected {expected_seq}")
            if prev_hash != prev:
                return ChainCheck(False, checked, seq, "prev_hash does not match preceding record")
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                return ChainCheck(False, checked, seq, "payload is not valid JSON")
            if record_hash(seq, etype, tx, vt, payload, prev_hash) != h:
                return ChainCheck(False, checked, seq, "record hash mismatch")
            prev = h
            expected_seq += 1
            checked += 1
        return ChainCheck(True, checked)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AuditLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
