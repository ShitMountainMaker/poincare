import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.hyperbolic_utils import (
    pairwise_poincare_distance,
    project_to_poincare_ball,
)


def pairwise_lcp_matrix(cluster_ids: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise longest common prefix length for semantic ids."""
    if cluster_ids.dim() != 2:
        raise ValueError("cluster_ids must have shape [batch_size, num_hierarchies].")

    equality_matrix = cluster_ids.unsqueeze(1).eq(cluster_ids.unsqueeze(0))
    prefix_matches = equality_matrix.to(dtype=torch.int64).cumprod(dim=-1)
    return prefix_matches.sum(dim=-1)


def pairwise_squared_euclidean_distance(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute a pairwise squared Euclidean distance matrix."""
    sq_norm = embeddings.pow(2).sum(dim=-1, keepdim=True)
    return (
        sq_norm + sq_norm.transpose(0, 1) - 2.0 * embeddings @ embeddings.transpose(0, 1)
    ).clamp_min(0.0)


def upper_triangle_pair_mask(size: int, device: torch.device) -> torch.Tensor:
    """Return a mask for unique non-diagonal pairs."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


class EuclideanPrefixLoss(nn.Module):
    """Match pairwise distances to prefix similarity in Euclidean space."""

    def __init__(
        self,
        normalize_embeddings: bool = True,
        max_normalized_distance: float = 4.0,
    ) -> None:
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.max_normalized_distance = max_normalized_distance

    def forward(self, embeddings: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        batch_size, num_hierarchies = cluster_ids.shape
        if batch_size < 2:
            return embeddings.new_tensor(0.0)

        working_embeddings = embeddings.float()
        if self.normalize_embeddings:
            working_embeddings = F.normalize(working_embeddings, dim=-1)

        lcp = pairwise_lcp_matrix(cluster_ids.long()).float()
        lcp_ratio = lcp / max(num_hierarchies, 1)

        pairwise_sq_dist = pairwise_squared_euclidean_distance(working_embeddings)
        if self.normalize_embeddings:
            target_dist = self.max_normalized_distance * (1.0 - lcp_ratio)
        else:
            target_dist = 1.0 - lcp_ratio

        pair_mask = upper_triangle_pair_mask(batch_size, device=embeddings.device)
        return F.mse_loss(
            pairwise_sq_dist[pair_mask],
            target_dist[pair_mask],
            reduction="mean",
        )


class HyperbolicPrefixContrastiveLoss(nn.Module):
    """A minimal hyperbolic prefix-aware regularizer with a small projector."""

    def __init__(
        self,
        input_dim: int,
        projector_dim: int = 64,
        negative_margin: float = 1.0,
        max_ball_norm: float = 0.99,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
        )
        self.negative_margin = negative_margin
        self.max_ball_norm = max_ball_norm
        self.eps = eps

    def forward(self, embeddings: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        batch_size, num_hierarchies = cluster_ids.shape
        if batch_size < 2:
            return embeddings.new_tensor(0.0)

        projected_embeddings = self.projector(embeddings.float())
        projected_embeddings = project_to_poincare_ball(
            projected_embeddings,
            max_norm=self.max_ball_norm,
            eps=self.eps,
        )

        lcp = pairwise_lcp_matrix(cluster_ids.long()).float()
        lcp_ratio = lcp / max(num_hierarchies, 1)
        pairwise_dist = pairwise_poincare_distance(projected_embeddings, eps=self.eps)

        positive_weights = lcp_ratio
        negative_weights = 1.0 - lcp_ratio
        pair_mask = upper_triangle_pair_mask(batch_size, device=embeddings.device)

        positive_loss = positive_weights * pairwise_dist.pow(2)
        negative_loss = negative_weights * F.relu(
            self.negative_margin - pairwise_dist
        ).pow(2)
        return (positive_loss[pair_mask] + negative_loss[pair_mask]).mean()
