"""Dataset validation, loading, and performer-disjoint splitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class MotionRecord:
    motion: torch.Tensor
    category: int
    performer: int
    sequence_id: str


def load_motion_archive(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the public repository's NPZ interchange format."""
    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Motion archive not found: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {"motions", "categories", "performers", "sequence_ids"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Archive is missing keys: {sorted(missing)}")
        data = {key: archive[key] for key in required}
    motions = data["motions"]
    if motions.ndim != 4 or motions.shape[-1] != 3:
        raise ValueError("motions must have shape (N, T, J, 3)")
    count = motions.shape[0]
    if any(data[key].shape[0] != count for key in required - {"motions"}):
        raise ValueError("All archive arrays must share the same first dimension")
    if not np.isfinite(motions).all():
        raise ValueError("motions contains NaN or infinite coordinates")
    if len(np.unique(data["sequence_ids"].astype(str))) != count:
        raise ValueError("sequence_ids must be unique")
    return data


class MotionDataset(Dataset[MotionRecord]):
    """Torch dataset backed by an in-memory, validated NPZ archive."""

    def __init__(self, path: str | Path, sequence_ids: set[str] | None = None) -> None:
        data = load_motion_archive(path)
        ids = data["sequence_ids"].astype(str)
        if sequence_ids is None:
            indices = np.arange(len(ids))
        else:
            indices = np.flatnonzero(np.isin(ids, list(sequence_ids)))
            missing = sequence_ids.difference(ids[indices].tolist())
            if missing:
                raise ValueError(f"Unknown sequence IDs: {sorted(missing)[:5]}")
        self.motions = torch.from_numpy(data["motions"][indices].astype(np.float32))
        self.categories = torch.from_numpy(data["categories"][indices].astype(np.int64))
        self.performers = torch.from_numpy(data["performers"][indices].astype(np.int64))
        self.sequence_ids = ids[indices].tolist()

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __getitem__(self, index: int) -> MotionRecord:
        return MotionRecord(
            motion=self.motions[index],
            category=int(self.categories[index]),
            performer=int(self.performers[index]),
            sequence_id=self.sequence_ids[index],
        )


def collate_motion_records(records: list[MotionRecord]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "motion": torch.stack([record.motion for record in records]),
        "category": torch.tensor([record.category for record in records], dtype=torch.long),
        "performer": torch.tensor([record.performer for record in records], dtype=torch.long),
        "sequence_id": [record.sequence_id for record in records],
    }


def performer_disjoint_split(
    performers: np.ndarray,
    sequence_ids: np.ndarray,
    ratios: tuple[float, float, float] = (0.60, 0.15, 0.25),
    seed: int = 7,
) -> dict[str, list[str]]:
    """Allocate complete performers while approximating sequence-count targets.

    A greedy assignment minimizes the normalized deficit to the requested target after a
    seeded shuffle. Every performer appears in exactly one partition.
    """
    if len(performers) != len(sequence_ids):
        raise ValueError("performers and sequence_ids must have equal length")
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must contain three positive values")
    ratio_sum = sum(ratios)
    targets = np.asarray(ratios, dtype=np.float64) / ratio_sum * len(sequence_ids)
    unique, counts = np.unique(performers, return_counts=True)
    if len(unique) < 3:
        raise ValueError("At least three performers are required for disjoint partitions")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    grouped = sorted(
        [(int(unique[i]), int(counts[i])) for i in order],
        key=lambda item: item[1],
        reverse=True,
    )
    names = ("train", "calibration", "test")
    assigned: dict[str, list[int]] = {name: [] for name in names}
    totals = np.zeros(3, dtype=np.float64)
    for position, (performer, count) in enumerate(grouped):
        if position < 3:
            choice = position
        else:
            deficits = (targets - totals) / np.maximum(targets, 1.0)
            choice = int(np.argmax(deficits))
        assigned[names[choice]].append(performer)
        totals[choice] += count
    result: dict[str, list[str]] = {}
    ids_as_str = sequence_ids.astype(str)
    for name in names:
        mask = np.isin(performers, assigned[name])
        result[name] = sorted(ids_as_str[mask].tolist())
    return result


def write_split(path: str | Path, split: dict[str, list[str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_split(path: str | Path) -> dict[str, list[str]]:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"train", "calibration", "test"}
    if set(split) != required:
        raise ValueError(f"Split file must contain exactly {sorted(required)}")
    seen: set[str] = set()
    for name in sorted(required):
        current = set(split[name])
        if overlap := seen.intersection(current):
            raise ValueError(f"Sequence leakage into {name}: {sorted(overlap)[:5]}")
        seen.update(current)
    return split
