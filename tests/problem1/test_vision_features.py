import unittest

import numpy as np

from src.problem1.feature_extraction.vision.face_features import full_feature_dimension
from src.problem1.feature_extraction.vision.quality import visual_quality


class VisionFeatureTest(unittest.TestCase):
    def test_competition_feature_dimension(self):
        self.assertEqual(full_feature_dimension(768), 871)

    def test_full_frame_quality_is_valid_without_face(self):
        image = np.full((64, 96, 3), 127, dtype=np.uint8)
        config = {
            "face_quality_base": 0.65,
            "scene_quality_base": 0.18,
            "quality_exposure_weight": 0.20,
            "quality_sharpness_weight": 0.15,
            "quality_sharpness_reference": 120.0,
        }
        score = visual_quality(image, (0, 0, 96, 64), False, config)
        self.assertGreater(score, 0.18)
        self.assertLessEqual(score, 0.53)


if __name__ == "__main__":
    unittest.main()
