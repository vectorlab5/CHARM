"""Metrics reported or used by the CHARM evaluation pipeline."""

from __future__ import annotations

import torch

from charm.density import ReferenceModel


def canonical_style_centrality(style: torch.Tensor, reference: ReferenceModel) -> torch.Tensor:
    """Return held-out empirical density ranks in [0, 1]."""
    return reference.score(style)


def foot_skate_rate(
    motion: torch.Tensor,
    contact: torch.Tensor,
    foot_indices: tuple[int, ...],
    horizontal_axes: tuple[int, int] = (0, 2),
    threshold: float = 0.02,
) -> torch.Tensor:
    """Fraction of contact transitions whose horizontal displacement exceeds a threshold."""
    if motion.ndim != 4:
        raise ValueError("motion must have shape (B,T,J,3)")
    if contact.shape != (motion.shape[0], motion.shape[1], len(foot_indices)):
        raise ValueError("contact must have shape (B,T,len(foot_indices))")
    feet = motion[:, :, list(foot_indices), :]
    velocity = feet[:, 1:, :, list(horizontal_axes)] - feet[:, :-1, :, list(horizontal_axes)]
    displacement = velocity.norm(dim=-1)
    planted = contact[:, 1:].bool() & contact[:, :-1].bool()
    denominator = planted.sum(dim=(1, 2)).clamp_min(1)
    slipping = (displacement > threshold) & planted
    return slipping.sum(dim=(1, 2)).float() / denominator.float()
