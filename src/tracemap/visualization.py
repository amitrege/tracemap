"""Small helpers for turning results into images."""

from __future__ import annotations

import torch

from .data import IMAGENET_MEAN, IMAGENET_STD


def normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """Scale a heatmap into the [0, 1] range."""
    heatmap = heatmap - heatmap.min()
    denom = heatmap.max().clamp_min(1e-8)
    return heatmap / denom


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization so the image looks normal again."""
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return torch.clamp(image * std + mean, 0.0, 1.0)


def overlay_heatmap(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    alpha: float = 0.35,
) -> torch.Tensor:
    """Blend a heatmap on top of an RGB image."""
    base_image = denormalize_image(image)
    heatmap = normalize_heatmap(heatmap)
    colorized = torch.stack(
        (
            heatmap,
            0.25 * heatmap,
            1.0 - heatmap,
        ),
        dim=0,
    )
    overlay = (1.0 - alpha) * base_image + alpha * colorized
    return torch.clamp(overlay, 0.0, 1.0)


def tensor_to_uint8_image(image: torch.Tensor) -> torch.Tensor:
    """Convert a float image tensor in [0, 1] to uint8 HWC form."""
    return (image.permute(1, 2, 0).clamp(0.0, 1.0) * 255).to(torch.uint8)
