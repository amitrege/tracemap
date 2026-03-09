"""Visualization helper tests."""

from __future__ import annotations

import torch

from tracemap.visualization import normalize_heatmap, overlay_heatmap


def test_overlay_heatmap_returns_rgb_image() -> None:
    """Overlaying a heatmap should keep the image shape and range."""
    image = torch.rand(3, 224, 224)
    heatmap = normalize_heatmap(torch.rand(224, 224))

    overlay = overlay_heatmap(image, heatmap)

    assert overlay.shape == torch.Size([3, 224, 224])
    assert float(overlay.min()) >= 0.0
    assert float(overlay.max()) <= 1.0
