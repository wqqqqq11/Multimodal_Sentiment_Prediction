"""Text evidence quality estimates."""

from __future__ import annotations

import numpy as np


def token_quality(tokens: list[dict[str, object]]) -> np.ndarray:
    values = []
    for item in tokens:
        token = str(item["token"])
        alnum = sum(ch.isalnum() for ch in token)
        values.append(0.55 + 0.45 * min(alnum / max(len(token), 1), 1.0))
    return np.asarray(values, dtype=np.float32)
