import math

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
    """Deep-level focused hyperbolic band loss over pairwise semantic relations."""

    def __init__(
        self,
        input_dim: int,
        projector_dim: int = 64,
        other_target_min_distance: float = 0.4,
        other_target_max_distance: float = 1.2,
        other_target_gamma: float = 1.5,
        other_bandwidth: float = 0.25,
        other_group_weight: float = 0.25,
        other_hard_fraction: float = 0.1,
        sibling_target_distance: float = 0.55,
        sibling_bandwidth: float = 0.08,
        sibling_group_weight: float = 1.5,
        sibling_lower_scale: float = 3.0,
        sibling_hard_fraction: float = 0.5,
        full_match_target_distance: float = 0.7,
        full_match_bandwidth: float = 0.1,
        full_match_group_weight: float = 2.0,
        full_match_lower_scale: float = 4.0,
        full_match_hard_fraction: float = 0.5,
        hard_min_pairs: int = 32,
        max_ball_norm: float = 0.99,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
        )
        self.other_target_min_distance = other_target_min_distance
        self.other_target_max_distance = other_target_max_distance
        self.other_target_gamma = other_target_gamma
        self.other_bandwidth = other_bandwidth
        self.other_group_weight = other_group_weight
        self.other_hard_fraction = other_hard_fraction
        self.sibling_target_distance = sibling_target_distance
        self.sibling_bandwidth = sibling_bandwidth
        self.sibling_group_weight = sibling_group_weight
        self.sibling_lower_scale = sibling_lower_scale
        self.sibling_hard_fraction = sibling_hard_fraction
        self.full_match_target_distance = full_match_target_distance
        self.full_match_bandwidth = full_match_bandwidth
        self.full_match_group_weight = full_match_group_weight
        self.full_match_lower_scale = full_match_lower_scale
        self.full_match_hard_fraction = full_match_hard_fraction
        self.hard_min_pairs = hard_min_pairs
        self.max_ball_norm = max_ball_norm
        self.eps = eps

    def _reduce_group_loss(
        self,
        loss_matrix: torch.Tensor,
        mask: torch.Tensor,
        hard_fraction: float,
    ) -> torch.Tensor:
        values = loss_matrix[mask]
        if values.numel() == 0:
            return loss_matrix.new_tensor(0.0)

        if 0.0 < hard_fraction < 1.0:
            keep_count = max(self.hard_min_pairs, math.ceil(values.numel() * hard_fraction))
            keep_count = min(values.numel(), keep_count)
            values = torch.topk(values, k=keep_count, largest=True).values
        return values.mean()

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
        exact_match_mask = lcp.eq(num_hierarchies)
        if num_hierarchies > 1:
            sibling_mask = lcp.eq(num_hierarchies - 1)
        else:
            sibling_mask = torch.zeros_like(exact_match_mask)

        pair_mask = upper_triangle_pair_mask(batch_size, device=embeddings.device)
        exact_match_mask = exact_match_mask & pair_mask
        sibling_mask = sibling_mask & pair_mask
        other_mask = pair_mask & ~(exact_match_mask | sibling_mask)
        lcp_ratio_others = lcp_ratio.clamp(
            max=max((num_hierarchies - 2) / max(num_hierarchies, 1), 0.0)
        )

        target_distance = self.other_target_min_distance + (
            (1.0 - lcp_ratio_others).pow(self.other_target_gamma)
            * (self.other_target_max_distance - self.other_target_min_distance)
        )
        bandwidth = torch.full_like(target_distance, self.other_bandwidth)
        lower_scale = torch.ones_like(target_distance)

        target_distance[sibling_mask] = self.sibling_target_distance
        bandwidth[sibling_mask] = self.sibling_bandwidth
        lower_scale[sibling_mask] = self.sibling_lower_scale

        target_distance[exact_match_mask] = self.full_match_target_distance
        bandwidth[exact_match_mask] = self.full_match_bandwidth
        lower_scale[exact_match_mask] = self.full_match_lower_scale

        a_ij = (target_distance - bandwidth).clamp_min(0.0)
        b_ij = target_distance + bandwidth
        lower_penalty = lower_scale * F.relu(a_ij - pairwise_dist).pow(2)
        upper_penalty = F.relu(pairwise_dist - b_ij).pow(2)
        band_loss = lower_penalty + upper_penalty

        weighted_group_losses = []
        group_weights = []

        if exact_match_mask.any():
            weighted_group_losses.append(
                self.full_match_group_weight
                * self._reduce_group_loss(
                    band_loss,
                    exact_match_mask,
                    self.full_match_hard_fraction,
                )
            )
            group_weights.append(self.full_match_group_weight)

        if sibling_mask.any():
            weighted_group_losses.append(
                self.sibling_group_weight
                * self._reduce_group_loss(
                    band_loss,
                    sibling_mask,
                    self.sibling_hard_fraction,
                )
            )
            group_weights.append(self.sibling_group_weight)

        if other_mask.any():
            weighted_group_losses.append(
                self.other_group_weight
                * self._reduce_group_loss(
                    band_loss,
                    other_mask,
                    self.other_hard_fraction,
                )
            )
            group_weights.append(self.other_group_weight)

        if not weighted_group_losses:
            return embeddings.new_tensor(0.0)
        return torch.stack(weighted_group_losses).sum() / sum(group_weights)
