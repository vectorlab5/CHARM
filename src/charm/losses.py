"""Losses for CHARM representation learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from charm.models.representation import CHARMRepresentation


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, inputs: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: object, gradients: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.weight * gradients, None


def gradient_reverse(inputs: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(inputs, weight)


class AdversarialClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor, reverse_weight: float = 0.0) -> torch.Tensor:
        if reverse_weight:
            inputs = gradient_reverse(inputs, reverse_weight)
        return self.network(inputs)


def cross_performer_supervised_contrastive_loss(
    style: torch.Tensor,
    categories: torch.Tensor,
    performers: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE with positives sharing category but not performer."""
    if style.shape[0] < 2:
        return style.sum() * 0.0
    normalized = F.normalize(style, dim=-1)
    logits = normalized @ normalized.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(style.shape[0], dtype=torch.bool, device=style.device)
    valid = ~eye
    positive = categories[:, None].eq(categories[None, :]) & performers[:, None].ne(
        performers[None, :]
    )
    exp_logits = torch.exp(logits) * valid
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive.sum(dim=1)
    usable = positive_count > 0
    if not usable.any():
        return style.sum() * 0.0
    mean_log_probability = (positive * log_probability).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_probability[usable].mean()


def swap_cycle_loss(
    model: CHARMRepresentation,
    style: torch.Tensor,
    execution: torch.Tensor,
) -> torch.Tensor:
    if style.shape[0] < 2:
        return style.sum() * 0.0
    permutation = torch.roll(torch.arange(style.shape[0], device=style.device), shifts=1)
    swapped_motion = model.decode(style[permutation], execution)
    recovered_style, recovered_execution = model.encode(swapped_motion)
    return F.mse_loss(recovered_style, style[permutation].detach()) + F.mse_loss(
        recovered_execution, execution.detach()
    )


@dataclass(slots=True)
class LossOutput:
    total: torch.Tensor
    reconstruction: torch.Tensor
    adversarial: torch.Tensor
    contrastive: torch.Tensor
    cycle: torch.Tensor


def representation_loss(
    model: CHARMRepresentation,
    motion: torch.Tensor,
    categories: torch.Tensor,
    performers: torch.Tensor,
    performer_adversary: AdversarialClassifier,
    category_adversary: AdversarialClassifier,
    lambda_adversarial: float = 1.0,
    lambda_contrastive: float = 0.5,
    lambda_cycle: float = 0.5,
    temperature: float = 0.07,
) -> LossOutput:
    reconstructed, style, execution = model(motion)
    reconstruction = F.mse_loss(reconstructed, motion)
    performer_logits = performer_adversary(style, reverse_weight=1.0)
    category_logits = category_adversary(execution, reverse_weight=1.0)
    adversarial = F.cross_entropy(performer_logits, performers) + F.cross_entropy(
        category_logits, categories
    )
    contrastive = cross_performer_supervised_contrastive_loss(
        style, categories, performers, temperature
    )
    cycle = swap_cycle_loss(model, style, execution)
    total = (
        reconstruction
        + lambda_adversarial * adversarial
        + lambda_contrastive * contrastive
        + lambda_cycle * cycle
    )
    return LossOutput(total, reconstruction, adversarial, contrastive, cycle)
