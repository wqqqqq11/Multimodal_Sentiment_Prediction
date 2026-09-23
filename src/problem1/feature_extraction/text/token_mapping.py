"""Token records used by the deterministic text encoder."""

from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)


def tokens_from_rows(text: str, rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return ordered token/span dictionaries, preferring audited preprocessing rows."""
    if rows:
        ordered = sorted(rows, key=lambda row: int(row["token_index"]))
        return [
            {"index": int(row["token_index"]), "token": str(row["token"]),
             "kind": str(row.get("token_kind", "word")), "start": int(row["char_start"]),
             "end": int(row["char_end"])}
            for row in ordered
        ]
    return [
        {"index": i, "token": match.group(), "kind": "word", "start": match.start(), "end": match.end()}
        for i, match in enumerate(_TOKEN_RE.finditer(text))
    ]
