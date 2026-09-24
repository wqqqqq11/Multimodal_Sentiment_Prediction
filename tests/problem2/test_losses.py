from pathlib import Path

import torch

from src.problem2.config import load_config
from src.problem2.data import Problem2Dataset, make_loader, reliability_features
from src.problem2.losses import distillation_loss, supervised_loss
from src.problem2.model import MRCDNet


ROOT = Path(__file__).resolve().parents[2]


def test_losses_are_finite_and_differentiable() -> None:
    cfg = load_config(ROOT / "configs" / "problem2_model.yaml")
    cfg["model"].update({"text_pretrained": False, "text_hidden_dim": 64, "text_layers": 1, "text_intermediate_dim": 128})
    dataset = Problem2Dataset(ROOT / "datasets" / "preprocessed_data" / "problem2" / "train.npz")
    batch = next(iter(make_loader(dataset, 6, False, 0, False, 2026)))
    batch["modality_reliability"] = reliability_features(batch)
    model = MRCDNet(cfg["model"])
    student = model(batch)
    with torch.no_grad():
        teacher = model(batch)
    class_weights = torch.tensor(cfg["training"]["class_weights"])
    supervised, _ = supervised_loss(student, batch, class_weights, cfg["training"]["loss_weights"])
    distilled, _ = distillation_loss(student, teacher, cfg["training"]["distillation_temperature"], cfg["training"]["loss_weights"], batch)
    total = supervised + distilled
    assert torch.isfinite(total)
    total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
