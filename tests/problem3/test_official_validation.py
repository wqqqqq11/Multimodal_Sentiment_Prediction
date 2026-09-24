from __future__ import annotations

from data_progressing.problem3.official_validation import (
    EXPECTED_REVISION,
    EXPECTED_TOKENIZER,
)


def test_official_tokenizer_is_fully_pinned() -> None:
    assert EXPECTED_TOKENIZER == "google-bert/bert-base-uncased"
    assert len(EXPECTED_REVISION) == 40
    int(EXPECTED_REVISION, 16)

