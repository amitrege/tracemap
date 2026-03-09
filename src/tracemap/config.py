"""Settings for TraceMap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_CLASS_NAMES = (
    "Abyssinian",
    "Birman",
    "Persian",
    "Beagle",
    "Chihuahua",
)


@dataclass(slots=True)
class TraceMapConfig:
    """A small bundle of settings for training and explanation."""

    dataset_root: Path = Path("data")
    cache_dir: Path = Path(".cache")
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
    backbone_name: str = "resnet18"
    pretrained_weights: str | None = "DEFAULT"
    feature_layer: str = "layer4"
    image_size: int = 224
    resize_size: int = 256
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 15
    early_stopping_patience: int = 3
    val_fraction: float = 0.2
    damping: float = 1e-2
    top_k: int = 5
    device: str = "auto"
    random_seed: int = 17

    @property
    def cache_path(self) -> Path:
        """Default location for the saved head and index."""
        return self.cache_dir / "tracemap_bundle.pt"

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Map class names to integer ids."""
        return {name: idx for idx, name in enumerate(self.class_names)}
