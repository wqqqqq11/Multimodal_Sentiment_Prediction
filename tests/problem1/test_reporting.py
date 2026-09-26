import unittest

from src.problem1.reporting import FEATURE_SUMMARY_FIELDS, build_feature_summary_rows


class FeatureSummaryTableTest(unittest.TestCase):
    def test_builds_compact_official_summary_row(self):
        sample_id = "sample-1"
        feature = [{
            "sample_id": sample_id,
            "status": "completed",
            "text_steps": "16",
            "text_dimension": "768",
            "audio_steps": "269",
            "audio_dimension": "768",
            "vision_steps": "28",
            "vision_dimension": "871",
        }]
        alignment = [{"sample_id": sample_id, "status": "completed", "consensus_steps": "12"}]
        audit = [{
            "sample_id": sample_id,
            "feature_files_complete": "True",
            "result_files_complete": "True",
            "mapping_audit_passed": "True",
            "padding_audit_passed": "True",
            "error_count": "0",
        }]
        rows = build_feature_summary_rows(
            [sample_id], feature, alignment, audit, {sample_id: {"duration_sec": 5.5}}
        )

        self.assertEqual(list(rows[0]), FEATURE_SUMMARY_FIELDS)
        self.assertEqual(rows[0]["模态类型"], "文本/音频/视频")
        self.assertEqual(rows[0]["原始有效时长"], 5.5)
        self.assertEqual(rows[0]["文本特征维度"], 768)
        self.assertEqual(rows[0]["音频特征维度"], 768)
        self.assertEqual(rows[0]["视频特征维度"], 871)
        self.assertEqual(rows[0]["对齐粒度"], 0.5)
        self.assertEqual(rows[0]["原始有效步数"], "16/269/28")
        self.assertEqual(rows[0]["对齐后有效步数"], 12)

    def test_failed_audit_blocks_official_summary(self):
        sample_id = "sample-1"
        feature = [{
            "sample_id": sample_id, "status": "resumed",
            "text_steps": 2, "text_dimension": 3,
            "audio_steps": 4, "audio_dimension": 5,
            "vision_steps": 6, "vision_dimension": 7,
        }]
        alignment = [{"sample_id": sample_id, "status": "resumed", "consensus_steps": 3}]
        audit = [{
            "sample_id": sample_id,
            "feature_files_complete": True,
            "result_files_complete": True,
            "mapping_audit_passed": False,
            "padding_audit_passed": True,
            "error_count": 1,
        }]
        with self.assertRaisesRegex(ValueError, "未通过"):
            build_feature_summary_rows(
                [sample_id], feature, alignment, audit, {sample_id: {"duration_sec": 1.2}}
            )

    def test_rejects_missing_feature_record(self):
        with self.assertRaisesRegex(ValueError, "特征清单缺失"):
            build_feature_summary_rows(
                ["sample-1"], [], [{"sample_id": "sample-1"}], [], {}
            )


if __name__ == "__main__":
    unittest.main()
