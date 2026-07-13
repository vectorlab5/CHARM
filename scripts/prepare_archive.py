"""Build the validated CHARM NPZ interchange archive from preprocessed NumPy arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from charm.data import load_motion_archive
from charm.preprocessing import (
    linear_resample,
    root_center,
    scale_by_bone,
    select_and_reorder_joints,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motions", required=True, help="Path to float array (N,T,J,3)")
    parser.add_argument("--categories", required=True, help="Path to integer array (N,)")
    parser.add_argument("--performers", required=True, help="Path to integer array (N,)")
    parser.add_argument("--sequence-ids", required=True, help="Path to string array (N,)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source", required=True, help="Dataset name and version")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--download-date", required=True, help="ISO date, for example 2026-07-13")
    parser.add_argument("--license-note", required=True, help="License or access-terms summary")
    parser.add_argument("--target-frames", type=int)
    parser.add_argument("--joint-map", help="JSON file containing target-ordered source indices")
    parser.add_argument("--root-joint", type=int, default=0)
    parser.add_argument("--no-root-center", action="store_true")
    parser.add_argument(
        "--scale-bone",
        help="Optional target-skeleton joint pair formatted SOURCE,TARGET",
    )
    parser.add_argument("--target-bone-length", type=float, default=1.0)
    args = parser.parse_args()

    motions = np.load(args.motions, allow_pickle=False).astype(np.float32)
    if motions.ndim != 4 or motions.shape[-1] != 3:
        raise ValueError("motions must have shape (N,T,J,3)")
    joint_map = None
    if args.joint_map:
        joint_map = json.loads(Path(args.joint_map).read_text(encoding="utf-8"))
        if not isinstance(joint_map, list):
            raise ValueError("joint-map JSON must contain a list of source joint indices")
    scale_bone = None
    if args.scale_bone:
        values = args.scale_bone.split(",")
        if len(values) != 2:
            raise ValueError("--scale-bone must be formatted SOURCE,TARGET")
        scale_bone = (int(values[0]), int(values[1]))
    processed = []
    scale_factors = []
    for sequence in motions:
        values = select_and_reorder_joints(sequence, joint_map) if joint_map else sequence.copy()
        if not args.no_root_center:
            values = root_center(values, args.root_joint)
        if scale_bone is not None:
            values, factor = scale_by_bone(values, scale_bone, args.target_bone_length)
            scale_factors.append(factor)
        if args.target_frames:
            values = linear_resample(values, args.target_frames)
        processed.append(values)
    motions = np.stack(processed).astype(np.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        motions=motions,
        categories=np.load(args.categories, allow_pickle=False).astype(np.int64),
        performers=np.load(args.performers, allow_pickle=False).astype(np.int64),
        sequence_ids=np.load(args.sequence_ids, allow_pickle=False).astype(str),
    )
    data = load_motion_archive(output)
    manifest = {
        "source": args.source,
        "source_url": args.source_url,
        "download_date": args.download_date,
        "license_note": args.license_note,
        "archive": output.name,
        "sequences": int(data["motions"].shape[0]),
        "frames": int(data["motions"].shape[1]),
        "joints": int(data["motions"].shape[2]),
        "categories": int(np.unique(data["categories"]).size),
        "performers": int(np.unique(data["performers"]).size),
        "preprocessing": {
            "joint_map": joint_map,
            "root_centered": not args.no_root_center,
            "root_joint": None if args.no_root_center else args.root_joint,
            "target_frames": args.target_frames,
            "scale_bone": scale_bone,
            "target_bone_length": args.target_bone_length if scale_bone else None,
            "scale_factor_median": float(np.median(scale_factors)) if scale_factors else None,
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
