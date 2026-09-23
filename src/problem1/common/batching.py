"""Small batching helpers for variable-length temporal tensors."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar
import numpy as np

T = TypeVar("T")


def chunks(values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size < 1:
        raise ValueError("batch size必须>=1")
    for start in range(0, len(values), size):
        yield values[start:start + size]


def pad_sequences(values: list[np.ndarray], maximum: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        raise ValueError("不能组批空序列")
    maximum = maximum or max(len(value) for value in values)
    output = np.zeros((len(values), maximum, values[0].shape[1]), dtype=values[0].dtype)
    mask = np.zeros((len(values), maximum), dtype=np.bool_)
    for index, value in enumerate(values):
        length = min(len(value), maximum)
        output[index, :length] = value[:length]
        mask[index, :length] = True
    return output, mask
