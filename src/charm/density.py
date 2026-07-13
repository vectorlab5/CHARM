"""Reference density models and held-out empirical rank calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class AffineCoupling(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dimension * 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = inputs * self.mask
        log_scale, shift = self.network(masked).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * (1.0 - self.mask)
        shift = shift * (1.0 - self.mask)
        outputs = masked + (1.0 - self.mask) * (inputs * torch.exp(log_scale) + shift)
        return outputs, log_scale.sum(dim=-1)

    def inverse(self, outputs: torch.Tensor) -> torch.Tensor:
        masked = outputs * self.mask
        log_scale, shift = self.network(masked).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * (1.0 - self.mask)
        shift = shift * (1.0 - self.mask)
        return masked + (1.0 - self.mask) * (outputs - shift) * torch.exp(-log_scale)


class RealNVP(nn.Module):
    """Small RealNVP model used for a category-specific style prior."""

    def __init__(self, dimension: int, layers: int = 8, hidden_dim: int = 256) -> None:
        super().__init__()
        masks = []
        for index in range(layers):
            mask = (torch.arange(dimension) + index) % 2
            masks.append(AffineCoupling(dimension, hidden_dim, mask.float()))
        self.dimension = dimension
        self.layers = nn.ModuleList(masks)

    def transform(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = inputs
        log_determinant = torch.zeros(inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
        for layer in self.layers:
            values, update = layer(values)
            log_determinant = log_determinant + update
        return values, log_determinant

    def inverse(self, latent: torch.Tensor) -> torch.Tensor:
        values = latent
        for layer in reversed(self.layers):
            values = layer.inverse(values)
        return values

    def log_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        latent, log_determinant = self.transform(inputs)
        base = -0.5 * (latent.square() + math.log(2.0 * math.pi)).sum(dim=-1)
        return base + log_determinant

    def sample(self, count: int, device: torch.device | str | None = None) -> torch.Tensor:
        target = device or next(self.parameters()).device
        latent = torch.randn(count, self.dimension, device=target)
        return self.inverse(latent)


@dataclass(slots=True)
class EmpiricalCDF:
    """Finite-sample rank map with the paper's add-one convention."""

    sorted_values: torch.Tensor

    @classmethod
    def fit(cls, calibration_values: torch.Tensor) -> EmpiricalCDF:
        if calibration_values.ndim != 1 or calibration_values.numel() == 0:
            raise ValueError("calibration_values must be a non-empty vector")
        return cls(torch.sort(calibration_values.detach().cpu()).values)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        flat = values.detach().cpu().reshape(-1)
        counts = torch.searchsorted(self.sorted_values, flat, right=True)
        scores = (1.0 + counts.float()) / (self.sorted_values.numel() + 1.0)
        return scores.reshape(values.shape).to(values.device)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"sorted_values": self.sorted_values}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> EmpiricalCDF:
        return cls(state["sorted_values"].detach().cpu())


@dataclass(slots=True)
class ReferenceModel:
    density: RealNVP
    calibration: EmpiricalCDF

    def score(self, style: torch.Tensor) -> torch.Tensor:
        return self.calibration(self.density.log_prob(style))


def fit_flow(
    flow: RealNVP,
    style_codes: torch.Tensor,
    epochs: int = 200,
    learning_rate: float = 2e-4,
    batch_size: int = 256,
) -> list[float]:
    """Fit a flow by maximum likelihood and return epoch losses."""
    if style_codes.ndim != 2 or style_codes.shape[0] < 2:
        raise ValueError("style_codes must have shape (N,D) with N >= 2")
    optimizer = torch.optim.AdamW(flow.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        permutation = torch.randperm(style_codes.shape[0], device=style_codes.device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(permutation), batch_size):
            batch = style_codes[permutation[start : start + batch_size]]
            loss = -flow.log_prob(batch).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        losses.append(epoch_loss / max(batches, 1))
    return losses
