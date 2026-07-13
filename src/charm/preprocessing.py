"""Auditable pose preprocessing primitives used by the CHARM data pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def select_and_reorder_joints(
    motion: np.ndarray,
    joint_indices: Sequence[int],
) -> np.ndarray:
    """Select source joints in the target skeleton order."""
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError("motion must have shape (T,J,3)")
    indices = np.asarray(joint_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("joint_indices must be a non-empty vector")
    if indices.min() < 0 or indices.max() >= motion.shape[1]:
        raise ValueError("joint_indices contains an out-of-range source joint")
    return motion[:, indices, :].copy()


def root_center(motion: np.ndarray, root_joint: int = 0) -> np.ndarray:
    """Express all joints relative to the root joint in every frame."""
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError("motion must have shape (T,J,3)")
    if not 0 <= root_joint < motion.shape[1]:
        raise ValueError("root_joint is outside the joint dimension")
    return motion - motion[:, root_joint : root_joint + 1, :]


def scale_by_bone(
    motion: np.ndarray,
    bone: tuple[int, int],
    target_length: float = 1.0,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Normalize metric scale using the median length of a stable reference bone."""
    source, target = bone
    if not (0 <= source < motion.shape[1] and 0 <= target < motion.shape[1]):
        raise ValueError("bone indices are outside the joint dimension")
    lengths = np.linalg.norm(motion[:, source] - motion[:, target], axis=-1)
    median = float(np.median(lengths))
    if not np.isfinite(median) or median <= epsilon:
        raise ValueError("reference bone has zero or invalid median length")
    factor = float(target_length) / median
    return motion * factor, factor


def linear_resample(motion: np.ndarray, target_frames: int) -> np.ndarray:
    """Linearly resample a motion sequence along time."""
    if motion.ndim != 3 or motion.shape[-1] != 3 or motion.shape[0] < 1:
        raise ValueError("motion must have shape (T,J,3) with T >= 1")
    if target_frames < 1:
        raise ValueError("target_frames must be positive")
    if motion.shape[0] == target_frames:
        return motion.copy()
    if motion.shape[0] == 1:
        return np.repeat(motion, target_frames, axis=0)
    source_time = np.linspace(0.0, 1.0, motion.shape[0])
    target_time = np.linspace(0.0, 1.0, target_frames)
    flattened = motion.reshape(motion.shape[0], -1)
    resampled = np.stack(
        [
            np.interp(target_time, source_time, flattened[:, index])
            for index in range(flattened.shape[1])
        ],
        axis=1,
    )
    return resampled.reshape(target_frames, motion.shape[1], 3).astype(motion.dtype)


def triangulate_dlt(
    points: np.ndarray,
    projection_matrices: np.ndarray,
    confidence: np.ndarray | None = None,
    minimum_views: int = 2,
) -> np.ndarray:
    """Triangulate `(T,V,J,2)` detections with calibrated 3x4 camera matrices.

    Missing detections may be encoded as non-finite coordinates or zero confidence. The
    function raises when fewer than ``minimum_views`` valid observations remain.
    """
    if points.ndim != 4 or points.shape[-1] != 2:
        raise ValueError("points must have shape (T,V,J,2)")
    if projection_matrices.shape != (points.shape[1], 3, 4):
        raise ValueError("projection_matrices must have shape (V,3,4)")
    if confidence is None:
        confidence = np.ones(points.shape[:-1], dtype=np.float64)
    if confidence.shape != points.shape[:-1]:
        raise ValueError("confidence must have shape (T,V,J)")
    output = np.empty((points.shape[0], points.shape[2], 3), dtype=np.float64)
    for frame in range(points.shape[0]):
        for joint in range(points.shape[2]):
            observations = points[frame, :, joint]
            weights = confidence[frame, :, joint]
            valid = np.isfinite(observations).all(axis=1) & np.isfinite(weights) & (weights > 0)
            if int(valid.sum()) < minimum_views:
                raise ValueError(f"Insufficient calibrated views at frame {frame}, joint {joint}")
            rows = []
            for view in np.flatnonzero(valid):
                x_coordinate, y_coordinate = observations[view]
                projection = projection_matrices[view]
                weight = float(np.sqrt(weights[view]))
                rows.append(weight * (x_coordinate * projection[2] - projection[0]))
                rows.append(weight * (y_coordinate * projection[2] - projection[1]))
            _, _, right_vectors = np.linalg.svd(np.stack(rows), full_matrices=False)
            homogeneous = right_vectors[-1]
            if abs(homogeneous[3]) < 1e-12:
                raise ValueError(f"Degenerate triangulation at frame {frame}, joint {joint}")
            output[frame, joint] = homogeneous[:3] / homogeneous[3]
    return output.astype(np.float32)
