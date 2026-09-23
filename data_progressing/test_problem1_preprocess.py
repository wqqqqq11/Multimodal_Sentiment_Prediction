"""问题一预处理管线的轻量单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from problem1_preprocess import (
    atomic_write_json,
    build_visual_timeline,
    normalize_id,
    normalize_text,
    tokenize_with_offsets,
)


class Problem1PreprocessTests(unittest.TestCase):
    def test_normalize_id_keeps_youtube_style_ids(self) -> None:
        self.assertEqual(normalize_id("-3g5yACwYnA"), "-3g5yACwYnA")
        self.assertEqual(normalize_id(3.0), "3")

    def test_text_normalization_and_offsets_are_reversible(self) -> None:
        text = normalize_text("  I\u00a0can't\tbelieve  it!  ")
        self.assertEqual(text, "I can't believe it!")
        tokens = tokenize_with_offsets(text)
        self.assertEqual([item["token"] for item in tokens], ["I", "can't", "believe", "it", "!"])
        for item in tokens:
            self.assertEqual(text[item["char_start"] : item["char_end"]], item["token"])

    def test_visual_timeline_is_monotonic_and_in_range(self) -> None:
        timestamps = [index / 25.0 for index in range(50)]
        rows = build_visual_timeline(
            sample_id="video$_$0",
            frame_timestamps_sec=timestamps,
            sample_fps=5.0,
        )
        self.assertEqual(len(rows), 10)
        frame_indices = [row["source_frame_index"] for row in rows]
        self.assertEqual(frame_indices, sorted(set(frame_indices)))
        self.assertTrue(all(0 <= index < 50 for index in frame_indices))
        self.assertTrue(all(row["start_sec"] < row["end_sec"] <= 2.0 for row in rows))
        self.assertTrue(
            all(
                row["source_timestamp_sec"] == timestamps[row["source_frame_index"]]
                for row in rows
            )
        )

    def test_visual_timeline_uses_decodable_timestamps_not_container_estimate(self) -> None:
        timestamps = [index / 30.0 for index in range(81)]
        rows = build_visual_timeline(
            sample_id="short-visual-stream$_$0",
            frame_timestamps_sec=timestamps,
            sample_fps=5.0,
        )
        self.assertEqual(len(rows), 14)
        self.assertTrue(all(row["source_frame_index"] < 81 for row in rows))
        self.assertLessEqual(rows[-1]["end_sec"], 81 / 30.0 + 1e-9)

    def test_visual_timeline_rejects_non_monotonic_timestamps(self) -> None:
        with self.assertRaises(ValueError):
            build_visual_timeline(
                sample_id="bad$_$0",
                frame_timestamps_sec=[0.0, 0.04, 0.04],
                sample_fps=5.0,
            )

    def test_atomic_json_rejects_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaises(ValueError):
                atomic_write_json(path, {"bad": np.nan})
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
