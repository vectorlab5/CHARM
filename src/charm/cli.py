"""Command-line entry points for the public CHARM workflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from charm.config import load_config, resolve_device, seed_everything
from charm.data import (
    MotionDataset,
    collate_motion_records,
    load_motion_archive,
    performer_disjoint_split,
    read_split,
    write_split,
)
from charm.density import EmpiricalCDF, RealNVP, fit_flow
from charm.losses import representation_loss
from charm.retrieval import RetrievalMemory
from charm.runtime import (
    build_adversaries,
    build_diffusion,
    build_representation,
    encode_dataset,
    load_checkpoint,
    save_checkpoint,
)

LOGGER = logging.getLogger("charm")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _config_and_device(path: str) -> tuple[dict[str, Any], torch.device]:
    config = load_config(path)
    seed_everything(int(config["seed"]))
    return config, resolve_device(str(config.get("device", "auto")))


def make_splits_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create performer-disjoint CHARM splits")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    data = load_motion_archive(args.archive)
    split = performer_disjoint_split(data["performers"], data["sequence_ids"], seed=args.seed)
    write_split(args.output, split)
    print(json.dumps({name: len(ids) for name, ids in split.items()}, sort_keys=True))


def train_representation_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train CHARM representation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--initialize-from",
        help="Optional version-1 representation checkpoint used for initialization",
    )
    args = parser.parse_args(argv)
    _configure_logging()
    config, device = _config_and_device(args.config)
    split = read_split(config["data"]["split_path"])
    dataset = MotionDataset(config["data"]["path"], set(split["train"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_motion_records,
    )
    model = build_representation(config).to(device)
    if args.initialize_from:
        initialization = load_checkpoint(args.initialize_from, "representation", device)
        model.load_state_dict(initialization["model"])
    performer_adversary, category_adversary = build_adversaries(config)
    performer_adversary = performer_adversary.to(device)
    category_adversary = category_adversary.to(device)
    parameters = (
        list(model.parameters())
        + list(performer_adversary.parameters())
        + list(category_adversary.parameters())
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = args.epochs or int(config["training"]["representation_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        totals = np.zeros(5, dtype=np.float64)
        batches = 0
        for batch in loader:
            motion = batch["motion"].to(device)
            category = batch["category"].to(device)
            performer = batch["performer"].to(device)
            output = representation_loss(
                model,
                motion,
                category,
                performer,
                performer_adversary,
                category_adversary,
                lambda_adversarial=float(config["loss"]["lambda_adversarial"]),
                lambda_contrastive=float(config["loss"]["lambda_contrastive"]),
                lambda_cycle=float(config["loss"]["lambda_cycle"]),
                temperature=float(config["loss"]["contrastive_temperature"]),
            )
            optimizer.zero_grad(set_to_none=True)
            output.total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(config["training"]["grad_clip"]))
            optimizer.step()
            totals += [
                float(output.total.detach()),
                float(output.reconstruction.detach()),
                float(output.adversarial.detach()),
                float(output.contrastive.detach()),
                float(output.cycle.detach()),
            ]
            batches += 1
        means = totals / max(batches, 1)
        scheduler.step()
        record = dict(
            zip(
                ("total", "reconstruction", "adversarial", "contrastive", "cycle"),
                means,
                strict=True,
            )
        )
        record["learning_rate"] = float(scheduler.get_last_lr()[0])
        history.append(record)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            LOGGER.info("epoch=%d total=%.6f", epoch + 1, record["total"])
    save_checkpoint(
        args.output,
        "representation",
        config,
        {
            "model": model.state_dict(),
            "performer_adversary": performer_adversary.state_dict(),
            "category_adversary": category_adversary.state_dict(),
            "history": history,
        },
    )


def fit_reference_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit category reference flows and CSCS maps")
    parser.add_argument("--config", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    _configure_logging()
    config, device = _config_and_device(args.config)
    checkpoint = load_checkpoint(args.representation, "representation", device)
    model = build_representation(config).to(device)
    model.load_state_dict(checkpoint["model"])
    split = read_split(config["data"]["split_path"])
    train = encode_dataset(
        model, MotionDataset(config["data"]["path"], set(split["train"])), device
    )
    calibration = encode_dataset(
        model, MotionDataset(config["data"]["path"], set(split["calibration"])), device
    )
    references: dict[int, dict[str, Any]] = {}
    for category in range(int(config["num_categories"])):
        train_mask = train["category"] == category
        calibration_mask = calibration["category"] == category
        if train_mask.sum() < 2 or calibration_mask.sum() < 1:
            raise ValueError(f"Insufficient train/calibration codes for category {category}")
        flow = RealNVP(
            int(config["model"]["style_dim"]),
            layers=int(config["reference"]["flow_layers"]),
            hidden_dim=int(config["reference"]["flow_hidden_dim"]),
        ).to(device)
        losses = fit_flow(
            flow,
            train["style"][train_mask].to(device),
            epochs=args.epochs or int(config["reference"]["flow_epochs"]),
            learning_rate=float(config["reference"]["flow_learning_rate"]),
        )
        with torch.no_grad():
            values = flow.log_prob(calibration["style"][calibration_mask].to(device))
        empirical = EmpiricalCDF.fit(values)
        references[category] = {
            "flow": {key: value.cpu() for key, value in flow.state_dict().items()},
            "calibration": empirical.state_dict(),
            "losses": losses,
        }
    save_checkpoint(args.output, "reference", config, {"references": references})


def train_generator_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train retrieval-conditioned motion diffusion")
    parser.add_argument("--config", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    _configure_logging()
    config, device = _config_and_device(args.config)
    representation_payload = load_checkpoint(args.representation, "representation", device)
    representation = build_representation(config).to(device)
    representation.load_state_dict(representation_payload["model"])
    representation.eval()
    split = read_split(config["data"]["split_path"])
    dataset = MotionDataset(config["data"]["path"], set(split["train"]))
    encoded = encode_dataset(representation, dataset, device)
    memory = RetrievalMemory(encoded["style"].to(device), encoded["category"].to(device))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_motion_records,
    )
    diffusion = build_diffusion(config).to(device)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = args.epochs or int(config["training"]["generator_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history: list[float] = []
    for epoch in range(epochs):
        total = 0.0
        batches = 0
        for batch in loader:
            motion = batch["motion"].to(device)
            category = batch["category"].to(device)
            with torch.no_grad():
                style, _ = representation.encode(motion)
                retrieved, _ = memory.query(
                    style, category, k=int(config["reference"]["retrieval_k"])
                )
            loss = diffusion.training_loss(motion, style, retrieved)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                diffusion.parameters(), float(config["training"]["grad_clip"])
            )
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        scheduler.step()
        history.append(total / max(batches, 1))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            LOGGER.info("epoch=%d diffusion=%.6f", epoch + 1, history[-1])
    save_checkpoint(
        args.output,
        "generator",
        config,
        {"diffusion": diffusion.state_dict(), "history": history},
    )


def evaluate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out CSCS")
    parser.add_argument("--config", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config, device = _config_and_device(args.config)
    representation_payload = load_checkpoint(args.representation, "representation", device)
    reference_payload = load_checkpoint(args.reference, "reference", device)
    model = build_representation(config).to(device)
    model.load_state_dict(representation_payload["model"])
    split = read_split(config["data"]["split_path"])
    encoded = encode_dataset(
        model, MotionDataset(config["data"]["path"], set(split["test"])), device
    )
    rows: list[dict[str, Any]] = []
    all_scores: list[float] = []
    for category in range(int(config["num_categories"])):
        mask = encoded["category"] == category
        if not mask.any():
            continue
        state = reference_payload["references"][category]
        flow = RealNVP(
            int(config["model"]["style_dim"]),
            layers=int(config["reference"]["flow_layers"]),
            hidden_dim=int(config["reference"]["flow_hidden_dim"]),
        ).to(device)
        flow.load_state_dict(state["flow"])
        calibration = EmpiricalCDF.from_state_dict(state["calibration"])
        with torch.no_grad():
            scores = calibration(flow.log_prob(encoded["style"][mask].to(device))).cpu()
        all_scores.extend(scores.tolist())
        rows.append(
            {
                "category": category,
                "count": int(mask.sum()),
                "mean_cscs": float(scores.mean()),
                "std_cscs": float(scores.std(unbiased=False)),
            }
        )
    result = {
        "split": "test",
        "count": len(all_scores),
        "mean_cscs": float(np.mean(all_scores)) if all_scores else None,
        "std_cscs": float(np.std(all_scores)) if all_scores else None,
        "categories": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
