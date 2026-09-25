from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

import src.problem3.training as training_module
from src.problem3.config import load_config
from src.problem3.evaluation import exact_modality_shapley, local_leave_one_out
from src.problem3.losses import Problem3Loss
from src.problem3.model import SEPCNet
from src.problem3.pipeline import _export_deployment, _load_checkpoint
from src.problem3.prototypes import PrototypeMemory
from src.problem3.utils import json_safe, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def tiny_config() -> dict:
    cfg = json.loads((ROOT / "configs" / "problem3.yaml").read_text(encoding="utf-8"))
    cfg["model"].update({
        "text_pretrained": False, "vocab_size": 200, "max_steps": 6, "audio_dim": 4,
        "vision_dim": 3, "hidden_dim": 16, "attention_heads": 4, "intermediate_dim": 32,
        "context_layers": 1, "reasoning_layers": 1, "reasoning_tokens": 2, "top_k": 4,
        "text_hidden_dim": 16, "text_layers": 1, "dropout": 0.0,
    })
    cfg["training"]["class_weights"] = [1.0, 1.0, 1.0]
    cfg["evaluation"]["stability_repeats"] = 0
    return cfg


def tiny_batch() -> dict[str, torch.Tensor]:
    batch = 2
    steps = 6
    content = torch.tensor([[0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0]], dtype=torch.bool)
    return {
        "input_ids": torch.tensor([[101, 11, 12, 13, 14, 102], [101, 21, 22, 23, 102, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=torch.bool),
        "token_type_ids": torch.zeros(batch, steps, dtype=torch.long),
        "content_mask": content,
        "padding_mask": ~torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=torch.bool),
        "audio": torch.randn(batch, steps, 4), "vision": torch.randn(batch, steps, 3),
        "text_evidence_candidate_mask": content.clone(),
        "audio_evidence_candidate_mask": content.clone(),
        "vision_evidence_candidate_mask": content.clone(),
        "classification_labels": torch.tensor([0, 2]),
        "regression_labels": torch.tensor([-1.2, 1.8]),
        "intensity_bin": torch.tensor([2, 5]),
    }


def test_single_problem3_yaml_and_config_is_valid() -> None:
    cfg = load_config(ROOT / "configs" / "problem3.yaml")
    assert cfg["model"]["top_k"] == 8
    assert not (ROOT / "configs" / "problem3_model.yaml").exists()


def test_sparse_forward_respects_candidate_mask_and_backpropagates() -> None:
    cfg = tiny_config()
    model = SEPCNet(cfg)
    batch = tiny_batch()
    output = model(batch)
    assert output["logits"].shape == (2, 3)
    assert output["selected_indices"].shape == (2, 4)
    selected_candidate = torch.gather(output["all_candidate_mask"].reshape(2, -1), 1, output["selected_indices"])
    assert bool(selected_candidate.all())
    loss, terms = Problem3Loss(cfg)(output, batch, complement=model.predict_complement(output))
    loss.backward()
    assert np.isfinite(terms["total"])
    assert model.salience[0].network[0].weight.grad is not None


def test_shapley_and_local_importance_are_normalized() -> None:
    cfg = tiny_config()
    model = SEPCNet(cfg).eval()
    output = model(tiny_batch())
    shapley = exact_modality_shapley(model, output)
    local = local_leave_one_out(model, output)
    assert torch.allclose(shapley["modality_contribution"].sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(local.sum(dim=1), torch.ones(2), atol=1e-5)


def test_topk_pads_without_using_invalid_evidence() -> None:
    cfg = tiny_config()
    cfg["model"]["top_k"] = 8
    model = SEPCNet(cfg).eval()
    batch = tiny_batch()
    batch["text_evidence_candidate_mask"][0].zero_()
    batch["audio_evidence_candidate_mask"][0].zero_()
    batch["vision_evidence_candidate_mask"][0].zero_()
    batch["text_evidence_candidate_mask"][0, 1] = True
    batch["audio_evidence_candidate_mask"][0, 2] = True
    output = model(batch)
    assert int(output["selected_valid"][0].sum()) == 2
    assert int(output["hard_mask"][0].sum()) == 2
    assert torch.isfinite(output["logits"]).all()


def test_prototype_memory_retrieves_same_modality() -> None:
    memory = PrototypeMemory(4, 0.1, [-2.75, -2, -1, 0, 1, 2, 2.75])
    arrays = {
        "embedding": np.eye(4, dtype=np.float32)[:3], "modality": np.asarray([0, 1, 2]),
        "class_label": np.asarray([0, 1, 2]), "intensity_bin": np.asarray([1, 3, 5]),
        "intensity_value": np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
    }
    metadata = [{"sample_id": str(index), "position": index} for index in range(3)]
    memory.set_entries(arrays, metadata)
    query = torch.tensor([[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]])
    modality = torch.tensor([[0, 1, 2]])
    result = memory.retrieve(query, modality)
    assert result["nearest_prototype_index"].tolist() == [[0, 1, 2]]


def test_half_precision_all_masked_prototype_query_stays_finite() -> None:
    memory = PrototypeMemory(4, 0.12, [-2.75, -2, -1, 0, 1, 2, 2.75]).half()
    arrays = {
        "embedding": np.eye(4, dtype=np.float32)[:3], "modality": np.asarray([0, 1, 2]),
        "class_label": np.asarray([0, 1, 2]), "intensity_bin": np.asarray([1, 3, 5]),
        "intensity_value": np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
    }
    memory.set_entries(arrays, [{"sample_id": str(i), "position": i} for i in range(3)])
    query = torch.randn(2, 150, 4, dtype=torch.float16)
    modality = torch.arange(3).view(1, 3, 1).expand(2, 3, 50).reshape(2, 150)
    mask = torch.zeros(2, 150, dtype=torch.bool)
    result = memory.retrieve(query, modality, mask)
    assert torch.isfinite(result["prototype_logits"]).all()
    assert torch.isfinite(result["prototype_regression"]).all()
    assert torch.isfinite(result["nearest_prototype_similarity"]).all()
    assert not bool(result["prototype_available"].any())


def test_json_exports_replace_nonfinite_values_with_null(tmp_path: Path) -> None:
    payload = {"nan": float("nan"), "positive_inf": float("inf"), "array": np.asarray([1.0, np.nan])}
    safe = json_safe(payload)
    assert safe == {"nan": None, "positive_inf": None, "array": [1.0, None]}
    path = tmp_path / "strict.jsonl"
    write_jsonl(path, [payload])
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["nan"] is None


def test_deployment_checkpoint_allows_optional_posthoc_prototype_memory(tmp_path: Path) -> None:
    cfg = tiny_config()
    model = SEPCNet(cfg)
    empty_memory = PrototypeMemory(16, 0.1, list(cfg["model"]["intensity_bin_centers"]))
    standalone_path = _export_deployment(model, empty_memory, cfg, tmp_path / "standalone")
    standalone = torch.load(standalone_path, map_location="cpu", weights_only=False)
    assert standalone["prototype_required"] is False
    assert standalone["prototype_ready"] is False

    arrays = {
        "embedding": np.eye(16, dtype=np.float32)[:3],
        "modality": np.asarray([0, 1, 2]),
        "class_label": np.asarray([0, 1, 2]),
        "intensity_bin": np.asarray([1, 3, 5]),
        "intensity_value": np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
    }
    metadata = [{"sample_id": str(i), "position": i, "source_split": "train"} for i in range(3)]
    empty_memory.set_entries(arrays, metadata)
    run_dir = tmp_path / "complete"
    checkpoint_path = _export_deployment(model, empty_memory, cfg, run_dir)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert state["checkpoint_stage"] == "best_any_stage"
    assert state["checkpoint_role"] == "deployment_sparse_evidence_model"
    assert state["prototype_required"] is False
    assert state["prototype_pair_validated"] is False
    assert state["prototype_count"] == 3
    assert len(state["prototype_sha256"]) == 64
    assert Path(state["prototype_path"]).is_file()

    restored_model = SEPCNet(cfg)
    restored_memory = PrototypeMemory(16, 0.1, list(cfg["model"]["intensity_bin_centers"]))
    _load_checkpoint(restored_model, restored_memory, checkpoint_path, torch.device("cpu"), logging.getLogger(__name__))
    assert restored_memory.ready
    assert restored_memory.metadata == metadata

    with Path(state["prototype_path"]).open("ab") as handle:
        handle.write(b"tampered")
    rejected_memory = PrototypeMemory(16, 0.1, list(cfg["model"]["intensity_bin_centers"]))
    _load_checkpoint(model, rejected_memory, checkpoint_path, torch.device("cpu"), logging.getLogger(__name__))
    assert not rejected_memory.ready


def test_loading_standalone_checkpoint_does_not_require_prototypes(tmp_path: Path) -> None:
    cfg = tiny_config()
    model = SEPCNet(cfg)
    checkpoint_path = tmp_path / "orphan" / "checkpoints" / "model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({"model_state": model.state_dict(), "config": cfg}, checkpoint_path)
    restored_memory = PrototypeMemory(16, 0.1, list(cfg["model"]["intensity_bin_centers"]))
    _load_checkpoint(model, restored_memory, checkpoint_path, torch.device("cpu"), logging.getLogger(__name__))
    assert not restored_memory.ready


def test_global_best_can_come_from_warmup_and_prototypes_are_posthoc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tiny_config()
    cfg["training"].update({
        "warmup_epochs": 1,
        "joint_epochs": 2,
        "text_freeze_epochs": 0,
        "patience": 2,
        "min_delta": 0.0,
        "amp": False,
    })
    model = SEPCNet(cfg)
    model.lifecycle_epoch = 0
    scores = iter([9.0, 1.0, 0.5])
    validation_memory_versions: list[str | None] = []

    def fake_train_epoch(model_arg: SEPCNet, *args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        model_arg.lifecycle_epoch += 1
        return {"total": 1.0}

    def fake_evaluate(*args: object, **kwargs: object) -> dict[str, dict[str, float]]:
        prototype_memory = kwargs.get("prototype_memory", args[4] if len(args) > 4 else None)
        validation_memory_versions.append(
            str(prototype_memory.metadata[0]["sample_id"]) if prototype_memory is not None else None
        )
        return {"metrics": {
            "selection_score": next(scores), "accuracy": 0.5, "macro_f1": 0.4,
            "mae": 0.8, "pearson": 0.3, "per_class_recall": [0.5, 0.2, 0.5],
        }}

    def fake_build(model_arg: object, loader: object, device: object, memory: PrototypeMemory,
                   prototypes_per_cell: int, minimum_confidence: float,
                   require_correct: bool = True) -> dict[str, int]:
        del loader, device, prototypes_per_cell, minimum_confidence, require_correct
        version = str(model_arg.lifecycle_epoch)
        arrays = {
            "embedding": np.eye(16, dtype=np.float32)[:3],
            "modality": np.asarray([0, 1, 2]),
            "class_label": np.asarray([0, 1, 2]),
            "intensity_bin": np.asarray([1, 3, 5]),
            "intensity_value": np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
        }
        metadata = [{"sample_id": version, "position": i, "source_split": "train"} for i in range(3)]
        memory.set_entries(arrays, metadata)
        return {"candidate_evidence": 3, "occupied_cells": 3, "prototypes": 3}

    monkeypatch.setattr(training_module, "_train_epoch", fake_train_epoch)
    monkeypatch.setattr(training_module, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(training_module, "build_prototype_memory", fake_build)
    result, memory = training_module.train_model(
        model, [], [], [], torch.device("cpu"), cfg, tmp_path, logging.getLogger(__name__)
    )
    checkpoint = torch.load(tmp_path / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert result["best_score"] == 9.0
    assert result["best_epoch"] == 1
    assert result["best_stage"] == "warmup"
    assert checkpoint["checkpoint_role"] == "best_performance_finalized"
    assert checkpoint["prototype_ready"] is True
    assert checkpoint["prototype_pair_validated"] is False
    assert len(checkpoint["prototype_sha256"]) == 64
    assert memory.ready
    assert validation_memory_versions == [None, None, None]
    assert memory.metadata[0]["sample_id"] == "3"
