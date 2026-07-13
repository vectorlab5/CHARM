"""Retrieval-conditioned motion diffusion with classifier-free guidance."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_time_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class MotionDenoiser(nn.Module):
    def __init__(
        self,
        num_joints: int,
        style_dim: int = 128,
        hidden_dim: int = 256,
        layers: int = 6,
        attention_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.motion_projection = nn.Linear(num_joints * 3, hidden_dim)
        self.style_projection = nn.Linear(style_dim, hidden_dim)
        self.retrieval_projection = nn.Linear(style_dim, hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Linear(hidden_dim, num_joints * 3)

    def forward(
        self,
        noisy_motion: torch.Tensor,
        timestep: torch.Tensor,
        style: torch.Tensor | None,
        retrieved: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_motion.ndim != 4 or noisy_motion.shape[2:] != (self.num_joints, 3):
            raise ValueError(f"Expected noisy_motion (B,T,{self.num_joints},3)")
        features = self.motion_projection(noisy_motion.flatten(start_dim=2))
        condition = self.time_projection(sinusoidal_time_embedding(timestep, self.hidden_dim))
        if style is not None:
            condition = condition + self.style_projection(style)
        if retrieved is not None:
            if retrieved.ndim != 3:
                raise ValueError("retrieved must have shape (B,K,D)")
            condition = condition + self.retrieval_projection(retrieved.mean(dim=1))
        encoded = self.transformer(features + condition.unsqueeze(1))
        return self.output(encoded).view_as(noisy_motion)


class MotionDiffusion(nn.Module):
    def __init__(
        self,
        denoiser: MotionDenoiser,
        steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        condition_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.steps = steps
        self.condition_dropout = condition_dropout
        beta = torch.linspace(beta_start, beta_end, steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def training_loss(
        self,
        clean_motion: torch.Tensor,
        style: torch.Tensor,
        retrieved: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = clean_motion.shape[0]
        timestep = torch.randint(0, self.steps, (batch,), device=clean_motion.device)
        noise = torch.randn_like(clean_motion)
        alpha_bar = self.alpha_bar[timestep].view(batch, 1, 1, 1)
        noisy = alpha_bar.sqrt() * clean_motion + (1.0 - alpha_bar).sqrt() * noise
        drop = torch.rand(batch, device=clean_motion.device) < self.condition_dropout
        conditioned_style = style.clone()
        conditioned_style[drop] = 0.0
        conditioned_retrieved = retrieved
        if retrieved is not None:
            conditioned_retrieved = retrieved.clone()
            conditioned_retrieved[drop] = 0.0
        predicted = self.denoiser(noisy, timestep, conditioned_style, conditioned_retrieved)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int, int],
        style: torch.Tensor,
        retrieved: torch.Tensor | None = None,
        guidance_weight: float = 3.0,
    ) -> torch.Tensor:
        if shape[0] != style.shape[0]:
            raise ValueError("Sample batch size must match style batch size")
        motion = torch.randn(shape, device=style.device)
        for step in reversed(range(self.steps)):
            timestep = torch.full((shape[0],), step, device=style.device, dtype=torch.long)
            conditional = self.denoiser(motion, timestep, style, retrieved)
            unconditional = self.denoiser(motion, timestep, None, None)
            predicted_noise = (
                1.0 + guidance_weight
            ) * conditional - guidance_weight * unconditional
            alpha = self.alpha[step]
            alpha_bar = self.alpha_bar[step]
            beta = self.beta[step]
            mean = (motion - (beta / torch.sqrt(1.0 - alpha_bar)) * predicted_noise) / torch.sqrt(
                alpha
            )
            if step > 0:
                motion = mean + torch.sqrt(beta) * torch.randn_like(motion)
            else:
                motion = mean
        return motion
