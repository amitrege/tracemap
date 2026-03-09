"""Head-only influence utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional


def compute_head_gradients(
    logits: torch.Tensor,
    labels: torch.Tensor,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """Compute one head gradient per example."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)

    probs = torch.softmax(logits, dim=-1)
    one_hot = functional.one_hot(labels, num_classes=logits.size(-1)).to(
        dtype=logits.dtype,
    )
    residual = probs - one_hot
    grad_weight = residual.unsqueeze(-1) * embeddings.unsqueeze(1)
    return torch.cat((grad_weight.flatten(start_dim=1), residual), dim=1)


def compute_batch_hessian(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """Compute the summed Hessian for the linear head."""
    probs = torch.softmax(logits, dim=-1)
    fisher_logits = torch.diag_embed(probs) - probs.unsqueeze(2) * probs.unsqueeze(1)
    augmented_embeddings = torch.cat(
        (
            embeddings,
            torch.ones(
                embeddings.size(0),
                1,
                device=embeddings.device,
                dtype=embeddings.dtype,
            ),
        ),
        dim=1,
    )
    feature_outer = (
        augmented_embeddings.unsqueeze(2) * augmented_embeddings.unsqueeze(1)
    )
    hessian = torch.einsum("bij,bkl->ikjl", fisher_logits, feature_outer)
    dim = augmented_embeddings.size(1) * logits.size(1)
    return hessian.reshape(dim, dim)


@dataclass(slots=True)
class HeadInfluenceIndex:
    """Cached tensors used to score training examples against a query."""

    train_embeddings: torch.Tensor
    train_labels: torch.Tensor
    train_gradients: torch.Tensor
    hessian: torch.Tensor
    hessian_inverse: torch.Tensor
    projected_train_gradients: torch.Tensor

    def score(self, query_gradient: torch.Tensor) -> torch.Tensor:
        """Score every indexed training example against one query gradient."""
        query_gradient = query_gradient.to(self.projected_train_gradients.dtype)
        return -(self.projected_train_gradients @ query_gradient)

    def to_state(self) -> dict[str, torch.Tensor]:
        """Turn the index into a plain tensor dict for saving."""
        return {
            "train_embeddings": self.train_embeddings.cpu(),
            "train_labels": self.train_labels.cpu(),
            "train_gradients": self.train_gradients.cpu(),
            "hessian": self.hessian.cpu(),
            "hessian_inverse": self.hessian_inverse.cpu(),
            "projected_train_gradients": self.projected_train_gradients.cpu(),
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, torch.Tensor],
        device: torch.device,
    ) -> HeadInfluenceIndex:
        """Rebuild an index from saved tensors."""
        return cls(
            train_embeddings=state["train_embeddings"].to(device),
            train_labels=state["train_labels"].to(device),
            train_gradients=state["train_gradients"].to(device),
            hessian=state["hessian"].to(device),
            hessian_inverse=state["hessian_inverse"].to(device),
            projected_train_gradients=state["projected_train_gradients"].to(device),
        )
