from pathlib import Path

import torch

from src.problem2.data import Problem2Dataset, apply_fixed_scenario, apply_random_mask_view, make_loader


ROOT = Path(__file__).resolve().parents[2]


def _batch():
    base = ROOT / "datasets" / "preprocessed_data" / "problem2"
    dataset = Problem2Dataset(base / "train.npz", base / "train_mask_bank.npz")
    return next(iter(make_loader(dataset, 8, False, 0, False, 2026)))


def test_random_mask_preserves_shape_and_zeros_missing_values() -> None:
    masked = apply_random_mask_view(_batch(), 1.0, torch.Generator().manual_seed(7))
    assert masked["modality_reliability"].shape == (8, 3, 8)
    for modality in ("audio", "vision"):
        missing = masked[f"{modality}_missing_mask"]
        assert torch.all(masked[modality][missing] == 0)


def test_fixed_scenario_increases_missingness() -> None:
    batch = _batch()
    masked = apply_fixed_scenario(batch, "audio_vision", 0.2, "middle")
    assert masked["audio_missing_mask"].sum() > batch["audio_missing_mask"].sum()
    assert masked["vision_missing_mask"].sum() > batch["vision_missing_mask"].sum()
