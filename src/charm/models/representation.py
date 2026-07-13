"""Style--execution representation used by CHARM."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dimension: int, maximum_length: int = 4096) -> None:
        super().__init__()
        positions = torch.arange(maximum_length).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10_000.0) / dimension))
        encoding = torch.zeros(maximum_length, dimension)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.encoding[: inputs.shape[1]].to(inputs.dtype)


def normalized_skeleton_adjacency(
    num_joints: int,
    edges: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Build a symmetric, degree-normalized skeleton adjacency with self-loops."""
    adjacency = torch.eye(num_joints)
    graph_edges = (
        edges if edges is not None else [(index, index + 1) for index in range(num_joints - 1)]
    )
    for edge in graph_edges:
        if len(edge) != 2:
            raise ValueError(f"Each skeleton edge must contain two indices, received {edge}")
        source, target = int(edge[0]), int(edge[1])
        if not (0 <= source < num_joints and 0 <= target < num_joints):
            raise ValueError(f"Skeleton edge {(source, target)} is outside [0, {num_joints})")
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


class SkeletonGraphBlock(nn.Module):
    """Residual graph convolution over a fixed joint topology."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.einsum("ij,...jh->...ih", adjacency, features)
        update = self.self_projection(features) + self.neighbor_projection(neighbors)
        return self.normalization(features + self.activation(update))


class MotionEncoder(nn.Module):
    """Graph-temporal encoder producing style and execution codes."""

    def __init__(
        self,
        num_joints: int,
        hidden_dim: int = 256,
        style_dim: int = 128,
        execution_dim: int = 128,
        transformer_layers: int = 4,
        attention_heads: int = 8,
        dropout: float = 0.1,
        skeleton_edges: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.register_buffer(
            "adjacency",
            normalized_skeleton_adjacency(num_joints, skeleton_edges),
            persistent=True,
        )
        self.joint_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.graph_blocks = nn.ModuleList([SkeletonGraphBlock(hidden_dim) for _ in range(2)])
        self.position = SinusoidalPositionEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.style_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, style_dim))
        self.execution_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, execution_dim)
        )

    def forward(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if motion.ndim != 4 or motion.shape[2] != self.num_joints or motion.shape[-1] != 3:
            raise ValueError(f"Expected (B,T,{self.num_joints},3), received {tuple(motion.shape)}")
        features = self.joint_projection(motion)
        for block in self.graph_blocks:
            features = block(features, self.adjacency)
        features = features.mean(dim=2)
        encoded = self.temporal(self.position(features))
        pooled = encoded.mean(dim=1)
        return self.style_head(pooled), self.execution_head(pooled)


class MotionDecoder(nn.Module):
    def __init__(
        self,
        num_joints: int,
        sequence_length: int,
        hidden_dim: int = 256,
        style_dim: int = 128,
        execution_dim: int = 128,
        transformer_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.1,
        skeleton_edges: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.sequence_length = sequence_length
        self.register_buffer(
            "adjacency",
            normalized_skeleton_adjacency(num_joints, skeleton_edges),
            persistent=True,
        )
        self.condition = nn.Linear(style_dim + execution_dim, hidden_dim)
        self.frame_queries = nn.Parameter(torch.randn(sequence_length, hidden_dim) * 0.02)
        self.joint_queries = nn.Parameter(torch.randn(num_joints, hidden_dim) * 0.02)
        self.position = SinusoidalPositionEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.graph_blocks = nn.ModuleList([SkeletonGraphBlock(hidden_dim) for _ in range(2)])
        self.output = nn.Linear(hidden_dim, 3)

    def forward(self, style: torch.Tensor, execution: torch.Tensor) -> torch.Tensor:
        if style.shape[0] != execution.shape[0]:
            raise ValueError("style and execution must have the same batch size")
        condition = self.condition(torch.cat([style, execution], dim=-1)).unsqueeze(1)
        queries = self.frame_queries.unsqueeze(0).expand(style.shape[0], -1, -1)
        decoded = self.temporal(self.position(queries + condition))
        decoded = decoded.unsqueeze(2) + self.joint_queries[None, None, :, :]
        for block in self.graph_blocks:
            decoded = block(decoded, self.adjacency)
        return self.output(decoded)


class CHARMRepresentation(nn.Module):
    def __init__(
        self,
        num_joints: int,
        sequence_length: int,
        hidden_dim: int = 256,
        style_dim: int = 128,
        execution_dim: int = 128,
        transformer_layers: int = 4,
        attention_heads: int = 8,
        dropout: float = 0.1,
        skeleton_edges: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = MotionEncoder(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            style_dim=style_dim,
            execution_dim=execution_dim,
            transformer_layers=transformer_layers,
            attention_heads=attention_heads,
            dropout=dropout,
            skeleton_edges=skeleton_edges,
        )
        self.decoder = MotionDecoder(
            num_joints=num_joints,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            style_dim=style_dim,
            execution_dim=execution_dim,
            attention_heads=attention_heads,
            dropout=dropout,
            skeleton_edges=skeleton_edges,
        )

    def encode(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(motion)

    def decode(self, style: torch.Tensor, execution: torch.Tensor) -> torch.Tensor:
        return self.decoder(style, execution)

    def forward(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        style, execution = self.encode(motion)
        return self.decode(style, execution), style, execution
