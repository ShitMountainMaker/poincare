import torch


def project_to_poincare_ball(
    embeddings: torch.Tensor,
    max_norm: float = 0.99,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project embeddings to the open Poincare ball."""
    if max_norm <= 0 or max_norm >= 1:
        raise ValueError("max_norm must be in the open interval (0, 1).")

    embeddings = embeddings.float()
    norms = torch.linalg.vector_norm(embeddings, dim=-1, keepdim=True).clamp_min(eps)
    clipped_max_norm = max_norm - eps
    scale = torch.clamp(clipped_max_norm / norms, max=1.0)
    return embeddings * scale


def pairwise_poincare_distance(
    embeddings: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute pairwise Poincare distances for embeddings inside the unit ball."""
    embeddings = embeddings.float()
    sq_norm = embeddings.pow(2).sum(dim=-1, keepdim=True).clamp(max=1 - eps)
    sq_dist = (
        sq_norm
        + sq_norm.transpose(0, 1)
        - 2.0 * embeddings @ embeddings.transpose(0, 1)
    ).clamp_min(0.0)
    denom = ((1.0 - sq_norm) * (1.0 - sq_norm.transpose(0, 1))).clamp_min(eps)
    acosh_argument = (1.0 + 2.0 * sq_dist / denom).clamp_min(1.0 + eps)
    return torch.acosh(acosh_argument)
