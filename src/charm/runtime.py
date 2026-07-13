"""Model construction, checkpoint I/O, and dataset encoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from charm.data import MotionDataset, collate_motion_records
from charm.losses import AdversarialClassifier
from charm.models.diffusion import MotionDenoiser, MotionDiffusion
from charm.models.representation import CHARMRepresentation


def build_representation(config: dict[str, Any]) -> CHARMRepresentation:
    data = config["data"]
    model = config["model"]
    return CHARMRepresentation(
        num_joints=int(data["num_joints"]),
        sequence_length=int(data["sequence_length"]),
        hidden_dim=int(model["hidden_dim"]),
        style_dim=int(model["style_dim"]),
        execution_dim=int(model["execution_dim"]),
        transformer_layers=int(model["transformer_layers"]),
        attention_heads=int(model["attention_heads"]),
        dropout=float(model["dropout"]),
        skeleton_edges=data.get("skeleton_edges"),
    )


def build_adversaries(
    config: dict[str, Any],
) -> tuple[AdversarialClassifier, AdversarialClassifier]:
    model = config["model"]
    return (
        AdversarialClassifier(int(model["style_dim"]), int(config["num_performers"])),
        AdversarialClassifier(int(model["execution_dim"]), int(config["num_categories"])),
    )


def build_diffusion(config: dict[str, Any]) -> MotionDiffusion:
    model = config["model"]
    diffusion = config["diffusion"]
    denoiser = MotionDenoiser(
        num_joints=int(config["data"]["num_joints"]),
        style_dim=int(model["style_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        attention_heads=int(model["attention_heads"]),
        dropout=float(model["dropout"]),
    )
    return MotionDiffusion(
        denoiser,
        steps=int(diffusion["steps"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
        condition_dropout=float(diffusion["condition_dropout"]),
    )


def save_checkpoint(
    path: str | Path,
    kind: str,
    config: dict[str, Any],
    state: dict[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "kind": kind, "config": config, **state}, output)


def load_checkpoint(path: str | Path, expected_kind: str, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format_version") != 1 or payload.get("kind") != expected_kind:
        raise ValueError(f"Expected a version-1 {expected_kind} checkpoint")
    return payload


@torch.no_grad()
def encode_dataset(
    model: CHARMRepresentation,
    dataset: MotionDataset,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, torch.Tensor | list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_motion_records,
    )
    model.eval()
    styles: list[torch.Tensor] = []
    executions: list[torch.Tensor] = []
    categories: list[torch.Tensor] = []
    performers: list[torch.Tensor] = []
    sequence_ids: list[str] = []
    for batch in loader:
        motion = batch["motion"].to(device)
        style, execution = model.encode(motion)
        styles.append(style.cpu())
        executions.append(execution.cpu())
        categories.append(batch["category"])
        performers.append(batch["performer"])
        sequence_ids.extend(batch["sequence_id"])
    return {
        "style": torch.cat(styles),
        "execution": torch.cat(executions),
        "category": torch.cat(categories),
        "performer": torch.cat(performers),
        "sequence_id": sequence_ids,
    }
