import unittest
from src.problem1.feature_extraction.text.token_mapping import tokens_from_rows


class TextFeatureTest(unittest.TestCase):
    def test_audited_token_mapping(self):
        rows = [{"token_index": "1", "token": "good", "token_kind": "word", "char_start": "5", "char_end": "9"},
                {"token_index": "0", "token": "Very", "token_kind": "word", "char_start": "0", "char_end": "4"}]
        tokens = tokens_from_rows("Very good", rows)
        self.assertEqual([item["token"] for item in tokens], ["Very", "good"])
        self.assertEqual(tokens[1]["start"], 5)


if __name__ == "__main__":
    unittest.main()
