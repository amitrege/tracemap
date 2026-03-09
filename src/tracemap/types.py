"""Result objects returned by TraceMap."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class Prediction:
    """Prediction details for the query image."""

    class_id: int
    class_name: str
    confidence: float


@dataclass(slots=True)
class PatchMatch:
    """One matched patch pair between the query and a training image."""

    query_x: int
    query_y: int
    train_x: int
    train_y: int
    similarity: float
    query_saliency: float
    train_saliency: float


@dataclass(slots=True)
class ExampleAttribution:
    """Everything TraceMap returns for one training example."""

    dataset_index: int
    class_id: int
    class_name: str
    influence_score: float
    affinity_score: float
    image: torch.Tensor
    heatmap: torch.Tensor
    query_heatmap: torch.Tensor
    patch_matches: list[PatchMatch] = field(default_factory=list)
    image_path: str | None = None


@dataclass(slots=True)
class ExplanationResult:
    """The full explanation for one query image."""

    prediction: Prediction
    query_image: torch.Tensor
    query_heatmap: torch.Tensor
    helpful_examples: list[ExampleAttribution]
    harmful_examples: list[ExampleAttribution]


@dataclass(slots=True)
class RemovalFaithfulnessCase:
    """Removal stats for one query."""

    query_dataset_index: int
    true_class_id: int
    true_class_name: str
    predicted_class_id: int
    predicted_class_name: str
    baseline_confidence: float
    helpful_removed_confidence: float
    random_removed_confidences: list[float] = field(default_factory=list)
    helpful_drop: float = 0.0
    random_mean_drop: float = 0.0
    helpful_beats_random: bool = False
    removed_train_indices: list[int] = field(default_factory=list)
    random_removed_indices: list[list[int]] = field(default_factory=list)


@dataclass(slots=True)
class RemovalFaithfulnessReport:
    """A short summary across removal-faithfulness runs."""

    removal_count: int
    random_trials: int
    mean_helpful_drop: float
    mean_random_drop: float
    win_rate: float
    cases: list[RemovalFaithfulnessCase] = field(default_factory=list)
