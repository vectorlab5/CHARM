"""Category-specific style-motif memory."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class RetrievalMemory:
    def __init__(self, style: torch.Tensor, category: torch.Tensor) -> None:
        if style.ndim != 2 or category.ndim != 1 or style.shape[0] != category.shape[0]:
            raise ValueError("style must be (N,D) and category must be (N,)")
        self.style = F.normalize(style.detach(), dim=-1)
        self.category = category.detach().long()

    def query(
        self, style: torch.Tensor, category: torch.Tensor, k: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if style.ndim != 2 or category.shape != (style.shape[0],):
            raise ValueError("Query style must be (B,D) with category shape (B,)")
        normalized = F.normalize(style, dim=-1)
        retrieved: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        for query, label in zip(normalized, category, strict=True):
            candidates = torch.nonzero(
                self.category == label,
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                raise ValueError(f"No retrieval entries for category {int(label)}")
            similarities = self.style[candidates] @ query
            count = min(k, candidates.numel())
            values, local_indices = torch.topk(similarities, count)
            selected = self.style[candidates[local_indices]]
            if count < k:
                pad = k - count
                selected = torch.cat([selected, selected[-1:].expand(pad, -1)], dim=0)
                values = torch.cat([values, values[-1:].expand(pad)], dim=0)
            retrieved.append(selected)
            scores.append(values)
        return torch.stack(retrieved), torch.stack(scores)
