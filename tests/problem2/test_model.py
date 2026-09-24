from pathlib import Path

import torch

from src.problem2.config import load_config
from src.problem2.data import Problem2Dataset, make_loader, reliability_features
from src.problem2.model import MRCDNet


ROOT = Path(__file__).resolve().parents[2]


def test_model_forward_uses_preprocessed_interface() -> None:
    cfg = load_config(ROOT / "configs" / "problem2_model.yaml")
    cfg["model"].update({"text_pretrained": False, "text_hidden_dim": 64, "text_layers": 1, "text_intermediate_dim": 128})
    dataset = Problem2Dataset(ROOT / "datasets" / "preprocessed_data" / "problem2" / "train.npz")
    loader = make_loader(dataset, 4, False, 0, False, 2026)
    batch = next(iter(loader))
    batch["modality_reliability"] = reliability_features(batch)
    outputs = MRCDNet(cfg["model"])(batch)
    assert outputs["logits"].shape == (4, 3)
    assert outputs["regression"].shape == (4,)
    assert outputs["gates"].shape == (4, 3)
    assert outputs["expert_logits"].shape == (4, 3, 3)
    assert outputs["text_tokens"].shape == (4, 50, 128)
    assert torch.allclose(outputs["gates"].sum(dim=1), torch.ones(4), atol=1e-5)
    assert torch.isfinite(outputs["regression"]).all()
