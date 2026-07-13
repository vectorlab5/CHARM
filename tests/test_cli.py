import json
from pathlib import Path

import numpy as np
import torch
import yaml

from charm.cli import (
    evaluate_main,
    fit_reference_main,
    train_generator_main,
    train_representation_main,
)


def test_one_epoch_pipeline_commands(tmp_path: Path) -> None:
    archive = tmp_path / "motions.npz"
    split_path = tmp_path / "splits.json"
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "representation.pt"
    reference_path = tmp_path / "reference.pt"
    generator_path = tmp_path / "generator.pt"
    metrics_path = tmp_path / "metrics.json"
    identifiers = np.asarray([f"sequence-{index}" for index in range(8)])
    np.savez_compressed(
        archive,
        motions=np.random.default_rng(5).normal(size=(8, 3, 2, 3)).astype(np.float32),
        categories=np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64),
        performers=np.asarray([0, 1, 0, 1, 2, 2, 3, 3], dtype=np.int64),
        sequence_ids=identifiers,
    )
    split_path.write_text(
        '{"train":["sequence-0","sequence-1","sequence-2","sequence-3"],'
        '"calibration":["sequence-4","sequence-5"],'
        '"test":["sequence-6","sequence-7"]}\n',
        encoding="utf-8",
    )
    config = {
        "seed": 5,
        "device": "cpu",
        "num_categories": 2,
        "num_performers": 4,
        "data": {
            "path": str(archive),
            "split_path": str(split_path),
            "num_joints": 2,
            "sequence_length": 3,
            "skeleton_edges": [[0, 1]],
        },
        "model": {
            "hidden_dim": 8,
            "style_dim": 4,
            "execution_dim": 4,
            "transformer_layers": 1,
            "attention_heads": 2,
            "dropout": 0.0,
        },
        "loss": {
            "lambda_adversarial": 1.0,
            "lambda_contrastive": 0.5,
            "lambda_cycle": 0.5,
            "contrastive_temperature": 0.07,
        },
        "reference": {
            "flow_layers": 2,
            "flow_hidden_dim": 8,
            "flow_epochs": 1,
            "flow_learning_rate": 0.0002,
            "retrieval_k": 2,
        },
        "diffusion": {
            "steps": 3,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "guidance_weight": 1.0,
            "condition_dropout": 0.0,
        },
        "training": {
            "representation_epochs": 1,
            "generator_epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.0002,
            "weight_decay": 0.01,
            "num_workers": 0,
            "grad_clip": 1.0,
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    train_representation_main(
        ["--config", str(config_path), "--output", str(checkpoint_path), "--epochs", "1"]
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["kind"] == "representation"
    assert len(payload["history"]) == 1
    fit_reference_main(
        [
            "--config",
            str(config_path),
            "--representation",
            str(checkpoint_path),
            "--output",
            str(reference_path),
            "--epochs",
            "1",
        ]
    )
    train_generator_main(
        [
            "--config",
            str(config_path),
            "--representation",
            str(checkpoint_path),
            "--output",
            str(generator_path),
            "--epochs",
            "1",
        ]
    )
    evaluate_main(
        [
            "--config",
            str(config_path),
            "--representation",
            str(checkpoint_path),
            "--reference",
            str(reference_path),
            "--output",
            str(metrics_path),
        ]
    )
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert result["split"] == "test"
    assert result["count"] == 2
    assert generator_path.is_file()
