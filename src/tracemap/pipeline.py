"""Main TraceMap pipeline."""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .config import TraceMapConfig
from .data import build_image_transform
from .influence import HeadInfluenceIndex, compute_batch_hessian, compute_head_gradients
from .model import FrozenResNet18Classifier
from .types import (
    ExampleAttribution,
    ExplanationResult,
    PatchMatch,
    Prediction,
    RemovalFaithfulnessCase,
    RemovalFaithfulnessReport,
)
from .visualization import normalize_heatmap


class TraceMap:
    """Fit the head, index the training set, and explain images."""

    def __init__(self, config: TraceMapConfig) -> None:
        """Set up the model and shared config."""
        self.config = config
        self.device = self._resolve_device(config.device)
        self.transform = build_image_transform(config)
        self.model = FrozenResNet18Classifier(
            num_classes=len(config.class_names),
            pretrained_weights=config.pretrained_weights,
        ).to(self.device)
        self.train_dataset: Dataset | None = None
        self.index: HeadInfluenceIndex | None = None
        self._fitted = False

    @classmethod
    def load_bundle(
        cls,
        path: str | Path,
        config: TraceMapConfig,
        train_dataset: Dataset | None = None,
    ) -> TraceMap:
        """Load a saved head and index back into a TraceMap instance."""
        pipeline = cls(config)
        state = torch.load(path, map_location=pipeline.device)
        pipeline.model.head.load_state_dict(state["head_state_dict"])
        pipeline.index = HeadInfluenceIndex.from_state(
            state["index"],
            device=pipeline.device,
        )
        pipeline.train_dataset = train_dataset
        pipeline._fitted = True
        return pipeline

    def save_bundle(self, path: str | Path | None = None) -> Path:
        """Save the head weights and influence index."""
        if not self._fitted or self.index is None:
            raise RuntimeError("Call fit() and build_index() before save_bundle().")

        output_path = Path(path or self.config.cache_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head_state_dict": self.model.head.state_dict(),
                "index": self.index.to_state(),
            },
            output_path,
        )
        return output_path

    def fit(self, train_dataset: Dataset, val_dataset: Dataset | None = None) -> None:
        """Train the linear head on frozen embeddings."""
        self._set_seed(self.config.random_seed)
        self.model.backbone.eval()

        train_embeddings, train_labels = self._extract_embeddings(train_dataset)
        val_embeddings = None
        val_labels = None
        if val_dataset is not None:
            val_embeddings, val_labels = self._extract_embeddings(val_dataset)

        trained_head = self._fit_head_from_embeddings(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            seed=self.config.random_seed,
        )
        self.model.head.load_state_dict(trained_head.state_dict())
        self.model.head.eval()
        self._fitted = True

    def build_index(self, train_dataset: Dataset) -> None:
        """Cache the tensors needed for influence scoring."""
        if not self._fitted:
            raise RuntimeError("Call fit() before build_index().")

        train_embeddings, train_labels = self._extract_embeddings(train_dataset)
        self.train_dataset = train_dataset

        logits = self.model.head(train_embeddings.to(self.device))
        gradients = compute_head_gradients(
            logits=logits,
            labels=train_labels.to(self.device),
            embeddings=train_embeddings.to(self.device),
        )
        hessian = compute_batch_hessian(
            logits=logits,
            embeddings=train_embeddings.to(self.device),
        ) / train_embeddings.size(0)
        hessian = hessian + self.config.damping * torch.eye(
            hessian.size(0),
            device=self.device,
            dtype=hessian.dtype,
        )

        hessian64 = hessian.to(torch.float64)
        cholesky_factor = torch.linalg.cholesky(hessian64)
        hessian_inverse = torch.cholesky_inverse(cholesky_factor).to(hessian.dtype)
        projected_gradients = gradients @ hessian_inverse

        self.index = HeadInfluenceIndex(
            train_embeddings=train_embeddings.to(self.device),
            train_labels=train_labels.to(self.device),
            train_gradients=gradients,
            hessian=hessian,
            hessian_inverse=hessian_inverse,
            projected_train_gradients=projected_gradients,
        )

    def explain(
        self,
        image: torch.Tensor | Image.Image | Sequence[torch.Tensor | Image.Image],
        target: int | str | Sequence[int | str | None] | None = None,
        top_k: int | None = None,
    ) -> ExplanationResult | list[ExplanationResult]:
        """Explain one image or a short batch of images."""
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            targets = self._expand_targets(target, image.size(0))
            return [
                self._explain_one(sample, sample_target, top_k)
                for sample, sample_target in zip(image, targets, strict=True)
            ]
        if isinstance(image, Sequence) and not isinstance(image, (str, bytes)):
            if image and isinstance(image[0], (torch.Tensor, Image.Image)):
                targets = self._expand_targets(target, len(image))
                return [
                    self._explain_one(sample, sample_target, top_k)
                    for sample, sample_target in zip(image, targets, strict=True)
                ]
        return self._explain_one(image, target, top_k)

    def evaluate_removal_faithfulness(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None,
        query_dataset: Dataset,
        num_queries: int = 6,
        removal_count: int = 3,
        random_trials: int = 3,
        top_k: int = 3,
    ) -> RemovalFaithfulnessReport:
        """Compare helpful-example removal against random removal."""
        if removal_count < 1:
            raise ValueError("removal_count must be at least 1.")
        if random_trials < 1:
            raise ValueError("random_trials must be at least 1.")

        if (
            not self._fitted
            or self.index is None
            or self.train_dataset is not train_dataset
        ):
            self.fit(train_dataset, val_dataset)
            self.build_index(train_dataset)

        train_embeddings, train_labels = self._extract_embeddings(train_dataset)
        val_embeddings = None
        val_labels = None
        if val_dataset is not None:
            val_embeddings, val_labels = self._extract_embeddings(val_dataset)

        query_embeddings, query_labels = self._extract_embeddings(query_dataset)
        with torch.no_grad():
            baseline_logits = self.model.head(query_embeddings.to(self.device))
            baseline_probabilities = torch.softmax(baseline_logits, dim=1).cpu()

        max_queries = min(num_queries, len(query_dataset), query_embeddings.size(0))
        cases: list[RemovalFaithfulnessCase] = []
        for query_index in range(max_queries):
            query_image, _ = query_dataset[query_index]
            predicted_class_id = int(
                baseline_probabilities[query_index].argmax().item(),
            )
            baseline_confidence = float(
                baseline_probabilities[query_index, predicted_class_id].item(),
            )

            explanation = self.explain(
                query_image,
                target=predicted_class_id,
                top_k=max(top_k, removal_count),
            )
            current_removal_count = min(
                removal_count,
                len(explanation.helpful_examples),
                max(len(train_dataset) - 1, 0),
            )
            if current_removal_count == 0:
                continue

            removed_train_indices = [
                example.dataset_index
                for example in explanation.helpful_examples[:current_removal_count]
            ]
            helpful_embeddings, helpful_labels = self._filter_embeddings(
                train_embeddings,
                train_labels,
                removed_train_indices,
            )
            retrain_seed = self.config.random_seed + 5_000 + query_index
            helpful_head = self._fit_head_from_embeddings(
                train_embeddings=helpful_embeddings,
                train_labels=helpful_labels,
                val_embeddings=val_embeddings,
                val_labels=val_labels,
                seed=retrain_seed,
            )
            helpful_removed_confidence = self._confidence_for_embedding(
                helpful_head,
                query_embeddings[query_index],
                predicted_class_id,
            )

            random_removed_confidences: list[float] = []
            random_removed_indices: list[list[int]] = []
            for trial in range(random_trials):
                rng = random.Random(
                    self.config.random_seed + query_index * 1_003 + trial,
                )
                sampled_indices = sorted(
                    rng.sample(range(len(train_dataset)), k=current_removal_count),
                )
                random_removed_indices.append(sampled_indices)
                random_embeddings, random_labels = self._filter_embeddings(
                    train_embeddings,
                    train_labels,
                    sampled_indices,
                )
                random_head = self._fit_head_from_embeddings(
                    train_embeddings=random_embeddings,
                    train_labels=random_labels,
                    val_embeddings=val_embeddings,
                    val_labels=val_labels,
                    seed=retrain_seed,
                )
                random_removed_confidences.append(
                    self._confidence_for_embedding(
                        random_head,
                        query_embeddings[query_index],
                        predicted_class_id,
                    ),
                )

            helpful_drop = baseline_confidence - helpful_removed_confidence
            random_drops = [
                baseline_confidence - confidence
                for confidence in random_removed_confidences
            ]
            random_mean_drop = sum(random_drops) / len(random_drops)

            true_class_id = int(query_labels[query_index].item())
            cases.append(
                RemovalFaithfulnessCase(
                    query_dataset_index=query_index,
                    true_class_id=true_class_id,
                    true_class_name=self.config.class_names[true_class_id],
                    predicted_class_id=predicted_class_id,
                    predicted_class_name=self.config.class_names[predicted_class_id],
                    baseline_confidence=baseline_confidence,
                    helpful_removed_confidence=helpful_removed_confidence,
                    random_removed_confidences=random_removed_confidences,
                    helpful_drop=helpful_drop,
                    random_mean_drop=random_mean_drop,
                    helpful_beats_random=helpful_drop > random_mean_drop,
                    removed_train_indices=removed_train_indices,
                    random_removed_indices=random_removed_indices,
                ),
            )

        if not cases:
            return RemovalFaithfulnessReport(
                removal_count=removal_count,
                random_trials=random_trials,
                mean_helpful_drop=0.0,
                mean_random_drop=0.0,
                win_rate=0.0,
                cases=[],
            )

        mean_helpful_drop = sum(case.helpful_drop for case in cases) / len(cases)
        mean_random_drop = sum(case.random_mean_drop for case in cases) / len(cases)
        win_rate = sum(case.helpful_beats_random for case in cases) / len(cases)
        return RemovalFaithfulnessReport(
            removal_count=removal_count,
            random_trials=random_trials,
            mean_helpful_drop=mean_helpful_drop,
            mean_random_drop=mean_random_drop,
            win_rate=win_rate,
            cases=cases,
        )

    def _explain_one(
        self,
        image: torch.Tensor | Image.Image,
        target: int | str | None,
        top_k: int | None,
    ) -> ExplanationResult:
        if self.index is None or self.train_dataset is None:
            raise RuntimeError("Call build_index() before explain().")

        query_image = self._prepare_image(image).to(self.device)
        query_batch = query_image.unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            forward = self.model(query_batch)
            probabilities = torch.softmax(forward.logits.squeeze(0), dim=0)

        predicted_class = int(probabilities.argmax().item())
        target_id = self._resolve_target_id(target, predicted_class)
        prediction = Prediction(
            class_id=predicted_class,
            class_name=self.config.class_names[predicted_class],
            confidence=float(probabilities[predicted_class].item()),
        )

        query_gradient = compute_head_gradients(
            logits=forward.logits,
            labels=torch.tensor([target_id], device=self.device),
            embeddings=forward.embeddings,
        ).squeeze(0)
        scores = self.index.score(query_gradient)
        k = min(top_k or self.config.top_k, len(scores))

        helpful_indices = torch.argsort(scores)[:k].tolist()
        harmful_indices = torch.argsort(scores, descending=True)[:k].tolist()
        helpful_examples = [
            self._build_example(
                query_image,
                target_id,
                train_index,
                float(scores[train_index].detach().cpu().item()),
            )
            for train_index in helpful_indices
        ]
        harmful_examples = [
            self._build_example(
                query_image,
                target_id,
                train_index,
                float(scores[train_index].detach().cpu().item()),
            )
            for train_index in harmful_indices
        ]
        aggregate_heatmap = self._aggregate_query_heatmap(
            helpful_examples if helpful_examples else harmful_examples,
        )

        return ExplanationResult(
            prediction=prediction,
            query_image=query_image.detach().cpu(),
            query_heatmap=aggregate_heatmap.detach().cpu(),
            helpful_examples=helpful_examples,
            harmful_examples=harmful_examples,
        )

    def _build_example(
        self,
        query_image: torch.Tensor,
        target_id: int,
        train_index: int,
        influence_score: float,
    ) -> ExampleAttribution:
        assert self.index is not None
        assert self.train_dataset is not None

        train_image, train_label = self.train_dataset[train_index]
        query_batch = query_image.unsqueeze(0).to(self.device)
        train_batch = train_image.unsqueeze(0).to(self.device)

        query_forward = self.model(query_batch, require_feature_grad=True)
        train_forward = self.model(train_batch, require_feature_grad=True)
        target_tensor = torch.tensor([target_id], device=self.device)
        train_label_tensor = torch.tensor([train_label], device=self.device)

        query_gradient = compute_head_gradients(
            logits=query_forward.logits,
            labels=target_tensor,
            embeddings=query_forward.embeddings,
        ).squeeze(0)
        train_gradient = compute_head_gradients(
            logits=train_forward.logits,
            labels=train_label_tensor,
            embeddings=train_forward.embeddings,
        ).squeeze(0)
        score = -(
            query_gradient.to(self.index.hessian_inverse.dtype)
            @ (
                self.index.hessian_inverse
                @ train_gradient.to(self.index.hessian_inverse.dtype)
            )
        )

        query_feature_grad, train_feature_grad = torch.autograd.grad(
            score,
            (query_forward.feature_map, train_forward.feature_map),
        )
        query_lowres = self._grad_cam(query_forward.feature_map, query_feature_grad)
        train_lowres = self._grad_cam(train_forward.feature_map, train_feature_grad)
        query_heatmap = self._upsample_heatmap(query_lowres, query_batch.shape[-2:])
        train_heatmap = self._upsample_heatmap(train_lowres, train_batch.shape[-2:])
        affinity_score = self._pair_affinity(
            query_forward.feature_map.detach(),
            query_lowres.detach(),
            train_forward.feature_map.detach(),
            train_lowres.detach(),
        )
        patch_matches = self._match_salient_patches(
            query_feature_map=query_forward.feature_map.detach(),
            query_heatmap=query_lowres.detach(),
            train_feature_map=train_forward.feature_map.detach(),
            train_heatmap=train_lowres.detach(),
            query_image_size=query_batch.shape[-2:],
            train_image_size=train_batch.shape[-2:],
        )

        image_path = None
        if hasattr(self.train_dataset, "get_image_path"):
            image_path = self.train_dataset.get_image_path(train_index)

        return ExampleAttribution(
            dataset_index=train_index,
            class_id=int(train_label),
            class_name=self.config.class_names[int(train_label)],
            influence_score=influence_score,
            affinity_score=affinity_score,
            image=train_image.detach().cpu(),
            heatmap=train_heatmap.detach().cpu(),
            query_heatmap=query_heatmap.detach().cpu(),
            patch_matches=patch_matches,
            image_path=image_path,
        )

    def _extract_embeddings(
        self,
        dataset: Dataset,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        embeddings: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        self.model.backbone.eval()
        self.model.head.eval()
        with torch.no_grad():
            for image_batch, label_batch in loader:
                image_batch = image_batch.to(self.device)
                _, embedding_batch = self.model.extract_features(image_batch)
                embeddings.append(embedding_batch.cpu())
                labels.append(label_batch.cpu())

        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

    def _fit_head_from_embeddings(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        val_embeddings: torch.Tensor | None,
        val_labels: torch.Tensor | None,
        seed: int,
    ) -> nn.Linear:
        """Train a fresh linear head from cached embeddings."""
        self._set_seed(seed)
        head = nn.Linear(
            train_embeddings.size(1),
            len(self.config.class_names),
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            TensorDataset(train_embeddings, train_labels),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        best_state = copy.deepcopy(head.state_dict())
        best_val_loss = float("inf")
        stale_epochs = 0
        for _ in range(self.config.num_epochs):
            head.train()
            for embedding_batch, label_batch in train_loader:
                embedding_batch = embedding_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                optimizer.zero_grad()
                logits = head(embedding_batch)
                loss = criterion(logits, label_batch)
                loss.backward()
                optimizer.step()

            if val_embeddings is None or val_labels is None:
                best_state = copy.deepcopy(head.state_dict())
                continue

            val_loss, _ = self._evaluate_head_module(head, val_embeddings, val_labels)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(head.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.early_stopping_patience:
                    break

        head.load_state_dict(best_state)
        head.eval()
        return head

    def _evaluate_head(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, float]:
        """Score the active head on cached embeddings."""
        return self._evaluate_head_module(self.model.head, embeddings, labels)

    def _evaluate_head_module(
        self,
        head: nn.Linear,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, float]:
        """Score any linear head on cached embeddings."""
        with torch.no_grad():
            logits = head(embeddings.to(self.device))
            loss = functional.cross_entropy(logits, labels.to(self.device)).item()
            accuracy = (
                (logits.argmax(dim=1) == labels.to(self.device)).float().mean().item()
            )
        return float(loss), float(accuracy)

    def _confidence_for_embedding(
        self,
        head: nn.Linear,
        embedding: torch.Tensor,
        class_id: int,
    ) -> float:
        """Return one class confidence for one embedding."""
        with torch.no_grad():
            logits = head(embedding.unsqueeze(0).to(self.device))
            probabilities = torch.softmax(logits.squeeze(0), dim=0)
        return float(probabilities[class_id].item())

    @staticmethod
    def _filter_embeddings(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        removed_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drop selected rows from cached embedding tensors."""
        keep_mask = torch.ones(embeddings.size(0), dtype=torch.bool)
        keep_mask[removed_indices] = False
        return embeddings[keep_mask], labels[keep_mask]

    def _prepare_image(self, image: torch.Tensor | Image.Image) -> torch.Tensor:
        if isinstance(image, Image.Image):
            return self.transform(image.convert("RGB"))
        if not isinstance(image, torch.Tensor):
            raise TypeError("Image must be a PIL image or a torch tensor.")
        if image.ndim != 3:
            raise ValueError("Single-image tensors must have shape [C, H, W].")
        return image

    def _resolve_target_id(self, target: int | str | None, default: int) -> int:
        if target is None:
            return default
        if isinstance(target, int):
            return target
        return self.config.class_to_idx[target]

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _grad_cam(feature_map: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * feature_map).sum(dim=1)).squeeze(0)
        return normalize_heatmap(heatmap)

    @staticmethod
    def _upsample_heatmap(
        heatmap: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        upsampled = functional.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        return normalize_heatmap(upsampled)

    def _match_salient_patches(
        self,
        query_feature_map: torch.Tensor,
        query_heatmap: torch.Tensor,
        train_feature_map: torch.Tensor,
        train_heatmap: torch.Tensor,
        query_image_size: tuple[int, int],
        train_image_size: tuple[int, int],
        max_matches: int = 4,
    ) -> list[PatchMatch]:
        """Return the strongest unique patch matches between two images."""
        query_tokens = query_feature_map.squeeze(0).flatten(start_dim=1).transpose(0, 1)
        train_tokens = train_feature_map.squeeze(0).flatten(start_dim=1).transpose(0, 1)
        query_weights = query_heatmap.flatten()
        train_weights = train_heatmap.flatten()

        query_count = max(1, int(query_weights.numel() * 0.15))
        train_count = max(1, int(train_weights.numel() * 0.15))
        query_indices = torch.topk(query_weights, k=query_count).indices
        train_indices = torch.topk(train_weights, k=train_count).indices

        selected_query = functional.normalize(query_tokens[query_indices], dim=1)
        selected_train = functional.normalize(train_tokens[train_indices], dim=1)
        similarities = selected_query @ selected_train.T

        used_query: set[int] = set()
        used_train: set[int] = set()
        matches: list[PatchMatch] = []
        num_train_candidates = len(train_indices)
        sorted_flat_indices = torch.argsort(
            similarities.flatten(),
            descending=True,
        ).tolist()
        for flat_index in sorted_flat_indices:
            query_rank = flat_index // num_train_candidates
            train_rank = flat_index % num_train_candidates
            if query_rank in used_query or train_rank in used_train:
                continue

            query_token_index = int(query_indices[query_rank].item())
            train_token_index = int(train_indices[train_rank].item())
            query_x, query_y = self._token_index_to_pixel_coordinates(
                token_index=query_token_index,
                token_grid=query_heatmap.shape,
                image_size=query_image_size,
            )
            train_x, train_y = self._token_index_to_pixel_coordinates(
                token_index=train_token_index,
                token_grid=train_heatmap.shape,
                image_size=train_image_size,
            )
            matches.append(
                PatchMatch(
                    query_x=query_x,
                    query_y=query_y,
                    train_x=train_x,
                    train_y=train_y,
                    similarity=float(similarities[query_rank, train_rank].item()),
                    query_saliency=float(query_weights[query_token_index].item()),
                    train_saliency=float(train_weights[train_token_index].item()),
                ),
            )
            used_query.add(query_rank)
            used_train.add(train_rank)
            if len(matches) >= max_matches:
                break

        return matches

    @staticmethod
    def _token_index_to_pixel_coordinates(
        token_index: int,
        token_grid: torch.Size,
        image_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Turn a flat token index into image-space coordinates."""
        grid_height, grid_width = int(token_grid[0]), int(token_grid[1])
        image_height, image_width = image_size
        token_y, token_x = divmod(token_index, grid_width)
        pixel_x = int(round(((token_x + 0.5) / grid_width) * image_width))
        pixel_y = int(round(((token_y + 0.5) / grid_height) * image_height))
        return (
            min(max(pixel_x, 0), image_width - 1),
            min(max(pixel_y, 0), image_height - 1),
        )

    @staticmethod
    def _pair_affinity(
        query_feature_map: torch.Tensor,
        query_heatmap: torch.Tensor,
        train_feature_map: torch.Tensor,
        train_heatmap: torch.Tensor,
    ) -> float:
        query_tokens = query_feature_map.squeeze(0).flatten(start_dim=1).transpose(0, 1)
        train_tokens = train_feature_map.squeeze(0).flatten(start_dim=1).transpose(0, 1)
        query_weights = query_heatmap.flatten()
        train_weights = train_heatmap.flatten()

        query_count = max(1, int(query_weights.numel() * 0.15))
        train_count = max(1, int(train_weights.numel() * 0.15))
        query_indices = torch.topk(query_weights, k=query_count).indices
        train_indices = torch.topk(train_weights, k=train_count).indices

        selected_query = functional.normalize(query_tokens[query_indices], dim=1)
        selected_train = functional.normalize(train_tokens[train_indices], dim=1)
        similarities = selected_query @ selected_train.T
        affinity = 0.5 * (
            similarities.max(dim=1).values.mean()
            + similarities.max(dim=0).values.mean()
        )
        return float(affinity.item())

    @staticmethod
    def _aggregate_query_heatmap(examples: list[ExampleAttribution]) -> torch.Tensor:
        if not examples:
            raise ValueError("Need at least one example to aggregate a query heatmap.")

        weight_tensor = torch.tensor([abs(item.influence_score) for item in examples])
        if float(weight_tensor.sum()) == 0.0:
            weight_tensor = torch.ones_like(weight_tensor)

        heatmaps = torch.stack([item.query_heatmap for item in examples], dim=0)
        aggregate = (
            (heatmaps * weight_tensor.view(-1, 1, 1)).sum(dim=0)
            / weight_tensor.sum()
        )
        return normalize_heatmap(aggregate)

    @staticmethod
    def _expand_targets(
        target: int | str | Sequence[int | str | None] | None,
        count: int,
    ) -> list[int | str | None]:
        if isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
            expanded = list(target)
            if len(expanded) != count:
                raise ValueError("Target sequence length must match the batch size.")
            return expanded
        return [target] * count
