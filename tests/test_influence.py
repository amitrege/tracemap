"""Unit tests for TraceMap's head-only influence math."""

from __future__ import annotations

import torch
from torch.nn import functional

from tracemap.influence import compute_batch_hessian, compute_head_gradients


def test_compute_head_gradients_matches_autograd() -> None:
    """The analytic head gradient should match autograd on a toy example."""
    embeddings = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
    labels = torch.tensor([1])
    weight = torch.tensor(
        [[0.1, 0.3], [-0.2, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    bias = torch.tensor([0.25, -0.1], dtype=torch.float64, requires_grad=True)

    logits = embeddings @ weight.transpose(0, 1) + bias
    loss = functional.cross_entropy(logits, labels)
    grad_weight, grad_bias = torch.autograd.grad(loss, (weight, bias))
    expected = torch.cat((grad_weight.flatten(), grad_bias))
    actual = compute_head_gradients(logits, labels, embeddings).squeeze(0)

    torch.testing.assert_close(actual, expected)


def test_compute_batch_hessian_matches_autograd() -> None:
    """The analytic head Hessian should match autograd on a toy example."""
    embeddings = torch.tensor([[0.5, -0.2]], dtype=torch.float64)
    labels = torch.tensor([0])
    params = torch.tensor(
        [0.2, -0.1, 0.3, 0.4, -0.2, 0.1],
        dtype=torch.float64,
        requires_grad=True,
    )

    def loss_fn(flat_params: torch.Tensor) -> torch.Tensor:
        weight = flat_params[:4].reshape(2, 2)
        bias = flat_params[4:]
        logits = embeddings @ weight.transpose(0, 1) + bias
        return functional.cross_entropy(logits, labels)

    expected = torch.autograd.functional.hessian(loss_fn, params)
    weight = params[:4].reshape(2, 2)
    bias = params[4:]
    logits = embeddings @ weight.transpose(0, 1) + bias
    actual = compute_batch_hessian(logits, embeddings)

    torch.testing.assert_close(actual, expected)
