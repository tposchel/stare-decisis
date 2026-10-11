"""Canonical serialization and hashing.

Every hash in the ledger is computed over canonical JSON: sorted keys, no
insignificant whitespace, UTF-8.  Enum members serialize as their values.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

GENESIS_HASH = "0" * 64


def _default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def canonical(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_default,
    )


def normalize(obj: Any) -> Any:
    """Round-trip through canonical JSON so in-memory payloads equal stored ones."""
    return json.loads(canonical(obj))


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    return sha256_hex(canonical(obj))
