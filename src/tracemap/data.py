"""Dataset loading helpers."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from .config import TraceMapConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class DatasetBundle:
    """The train, validation, and test splits used by the default setup."""

    train: Dataset
    val: Dataset | None
    test: Dataset


class OxfordPetSubsetDataset(Dataset):
    """A small Oxford-IIIT Pet subset with a fixed class list."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        class_names: tuple[str, ...],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        """Store the filtered image paths, labels, and transform."""
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.transform = transform

    @classmethod
    def from_torchvision(
        cls,
        root: Path,
        split: str,
        class_names: tuple[str, ...],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        download: bool = False,
    ) -> OxfordPetSubsetDataset:
        """Build the subset from torchvision's Oxford-IIIT Pet dataset."""
        base_dataset = OxfordIIITPet(
            root=str(root),
            split=split,
            target_types="category",
            download=download,
        )
        base_class_to_idx = {
            name: idx for idx, name in enumerate(base_dataset.classes)
        }
        missing = [name for name in class_names if name not in base_class_to_idx]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Unknown Oxford-IIIT Pet classes: {joined}")

        kept_ids = [base_class_to_idx[name] for name in class_names]
        remap = {class_id: idx for idx, class_id in enumerate(kept_ids)}

        image_paths: list[Path] = []
        labels: list[int] = []
        for image_path, raw_label in zip(
            base_dataset._images,
            base_dataset._labels,
            strict=True,
        ):
            if raw_label not in remap:
                continue
            image_paths.append(Path(image_path))
            labels.append(remap[raw_label])

        return cls(
            image_paths=image_paths,
            labels=labels,
            class_names=class_names,
            transform=transform,
        )

    def __len__(self) -> int:
        """Number of samples in the subset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load one transformed image tensor and its remapped label."""
        image = self.get_raw_image(index)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def get_raw_image(self, index: int) -> Image.Image:
        """Load the original PIL image for one item."""
        return Image.open(self.image_paths[index]).convert("RGB")

    def get_image_path(self, index: int) -> str:
        """Return the source path for one item."""
        return str(self.image_paths[index])

    def subset(self, indices: list[int]) -> OxfordPetSubsetDataset:
        """Make a smaller view with the same transform and class metadata."""
        return OxfordPetSubsetDataset(
            image_paths=[self.image_paths[idx] for idx in indices],
            labels=[self.labels[idx] for idx in indices],
            class_names=self.class_names,
            transform=self.transform,
        )


def build_image_transform(config: TraceMapConfig) -> transforms.Compose:
    """Build the image transform used everywhere in the pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(config.resize_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )


def stratified_train_val_split(
    dataset: OxfordPetSubsetDataset,
    val_fraction: float,
    seed: int,
) -> tuple[OxfordPetSubsetDataset, OxfordPetSubsetDataset | None]:
    """Split a dataset into train and validation pieces by class."""
    if val_fraction <= 0:
        return dataset, None

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    labels_t = dataset.labels
    for class_id in range(len(dataset.class_names)):
        class_indices = [idx for idx, label in enumerate(labels_t) if label == class_id]
        rng.shuffle(class_indices)
        if len(class_indices) <= 1:
            train_indices.extend(class_indices)
            continue
        val_count = max(1, int(len(class_indices) * val_fraction))
        val_count = min(val_count, len(class_indices) - 1)
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    train_indices.sort()
    val_indices.sort()
    return dataset.subset(train_indices), dataset.subset(val_indices)


def build_default_pet_datasets(
    config: TraceMapConfig,
    download: bool = False,
) -> DatasetBundle:
    """Build the default Oxford-IIIT Pet train, val, and test split."""
    transform = build_image_transform(config)
    trainval_dataset = OxfordPetSubsetDataset.from_torchvision(
        root=config.dataset_root,
        split="trainval",
        class_names=config.class_names,
        transform=transform,
        download=download,
    )
    train_dataset, val_dataset = stratified_train_val_split(
        trainval_dataset,
        val_fraction=config.val_fraction,
        seed=config.random_seed,
    )
    test_dataset = OxfordPetSubsetDataset.from_torchvision(
        root=config.dataset_root,
        split="test",
        class_names=config.class_names,
        transform=transform,
        download=download,
    )
    return DatasetBundle(train=train_dataset, val=val_dataset, test=test_dataset)
