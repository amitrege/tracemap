"""Integration-leaning tests for the TraceMap pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from tracemap import TraceMap, TraceMapConfig


class TinyImageDataset(Dataset):
    """Small synthetic image dataset for pipeline tests."""

    def __init__(self) -> None:
        """Initialize the synthetic images and labels."""
        self.images = torch.rand(8, 3, 224, 224)
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    def __len__(self) -> int:
        """Return the number of synthetic samples."""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return one synthetic image/label pair."""
        return self.images[index], int(self.labels[index].item())


def test_pipeline_fit_build_index_and_explain(tmp_path: Path) -> None:
    """TraceMap should fit, index, explain, and round-trip its bundle."""
    config = TraceMapConfig(
        class_names=("class_0", "class_1"),
        pretrained_weights=None,
        num_epochs=1,
        batch_size=2,
        top_k=2,
        cache_dir=tmp_path,
    )
    dataset = TinyImageDataset()

    pipeline = TraceMap(config)
    pipeline.fit(dataset)
    pipeline.build_index(dataset)
    result = pipeline.explain(dataset[0][0], top_k=2)

    assert result.prediction.class_name in config.class_names
    assert result.query_heatmap.shape == torch.Size([224, 224])
    assert len(result.helpful_examples) == 2
    assert len(result.harmful_examples) == 2
    assert 1 <= len(result.helpful_examples[0].patch_matches) <= 4

    bundle_path = pipeline.save_bundle()
    restored = TraceMap.load_bundle(bundle_path, config=config, train_dataset=dataset)
    restored_result = restored.explain(dataset[1][0], top_k=2)
    assert len(restored_result.helpful_examples) == 2
    assert len(restored_result.harmful_examples) == 2


def test_patch_matching_returns_unique_coordinates(tmp_path: Path) -> None:
    """Patch matching should emit bounded, unique query/train assignments."""
    config = TraceMapConfig(
        class_names=("class_0", "class_1"),
        pretrained_weights=None,
        cache_dir=tmp_path,
    )
    pipeline = TraceMap(config)

    query_feature_map = torch.rand(1, 4, 4, 4)
    train_feature_map = torch.rand(1, 4, 4, 4)
    query_heatmap = torch.linspace(0.0, 1.0, 16).reshape(4, 4)
    train_heatmap = torch.linspace(1.0, 0.0, 16).reshape(4, 4)

    matches = pipeline._match_salient_patches(
        query_feature_map=query_feature_map,
        query_heatmap=query_heatmap,
        train_feature_map=train_feature_map,
        train_heatmap=train_heatmap,
        query_image_size=(224, 224),
        train_image_size=(224, 224),
    )

    assert 1 <= len(matches) <= 4
    assert len({(match.query_x, match.query_y) for match in matches}) == len(matches)
    assert len({(match.train_x, match.train_y) for match in matches}) == len(matches)
    for match in matches:
        assert 0 <= match.query_x < 224
        assert 0 <= match.query_y < 224
        assert 0 <= match.train_x < 224
        assert 0 <= match.train_y < 224
        assert torch.isfinite(torch.tensor(match.similarity))


def test_removal_faithfulness_report_structure(tmp_path: Path) -> None:
    """Removal faithfulness should return per-query and aggregate metrics."""
    config = TraceMapConfig(
        class_names=("class_0", "class_1"),
        pretrained_weights=None,
        num_epochs=1,
        batch_size=2,
        top_k=2,
        cache_dir=tmp_path,
    )
    dataset = TinyImageDataset()

    pipeline = TraceMap(config)
    pipeline.fit(dataset)
    pipeline.build_index(dataset)
    report = pipeline.evaluate_removal_faithfulness(
        train_dataset=dataset,
        val_dataset=None,
        query_dataset=dataset,
        num_queries=2,
        removal_count=1,
        random_trials=2,
        top_k=1,
    )

    assert len(report.cases) == 2
    assert report.random_trials == 2
    assert torch.isfinite(torch.tensor(report.mean_helpful_drop))
    assert torch.isfinite(torch.tensor(report.mean_random_drop))
    for case in report.cases:
        assert len(case.random_removed_confidences) == 2
        assert len(case.random_removed_indices) == 2
        assert len(case.removed_train_indices) == 1
