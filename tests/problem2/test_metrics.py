import numpy as np

from src.problem2.metrics import all_metrics, confusion_matrix


def test_metrics_are_exact_for_perfect_predictions() -> None:
    classes = np.array([0, 1, 2, 0])
    regression = np.array([-2.0, 0.0, 2.0, -1.0])
    metrics = all_metrics(classes, classes.copy(), regression, regression.copy())
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["mae"] == 0.0
    assert np.isclose(metrics["pearson"], 1.0)
    assert confusion_matrix(classes, classes).trace() == 4
