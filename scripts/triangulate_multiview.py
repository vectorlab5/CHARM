"""Triangulate calibrated multi-view 2D poses into per-sequence 3D arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from charm.preprocessing import triangulate_dlt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, help="NumPy array shaped (N,T,V,J,2)")
    parser.add_argument("--cameras", required=True, help="Projection matrices shaped (V,3,4)")
    parser.add_argument("--confidence", help="Optional confidence array shaped (N,T,V,J)")
    parser.add_argument("--output", required=True, help="Output NumPy array shaped (N,T,J,3)")
    args = parser.parse_args()

    points = np.load(args.points, allow_pickle=False)
    cameras = np.load(args.cameras, allow_pickle=False)
    confidence = np.load(args.confidence, allow_pickle=False) if args.confidence else None
    if points.ndim != 5:
        raise ValueError("points must have shape (N,T,V,J,2)")
    if confidence is not None and confidence.shape != points.shape[:-1]:
        raise ValueError("confidence must have shape (N,T,V,J)")
    reconstructed = [
        triangulate_dlt(
            points[index],
            cameras,
            None if confidence is None else confidence[index],
        )
        for index in range(points.shape[0])
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.stack(reconstructed).astype(np.float32), allow_pickle=False)


if __name__ == "__main__":
    main()
