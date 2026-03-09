"""Model pieces used by TraceMap."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(slots=True)
class ForwardBatch:
    """The tensors returned by one forward pass."""

    logits: torch.Tensor
    embeddings: torch.Tensor
    feature_map: torch.Tensor


class FrozenResNet18Classifier(nn.Module):
    """A frozen ResNet18 with a trainable linear head."""

    def __init__(self, num_classes: int, pretrained_weights: str | None) -> None:
        """Build the backbone and head."""
        super().__init__()
        weights = None
        if pretrained_weights == "DEFAULT":
            weights = ResNet18_Weights.DEFAULT
        elif pretrained_weights is not None:
            weights = ResNet18_Weights[pretrained_weights]

        self.backbone = resnet18(weights=weights)
        self.embedding_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(self.embedding_dim, num_classes)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def extract_features(
        self,
        image_batch: torch.Tensor,
        require_feature_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the last conv map and pooled embedding."""
        if require_feature_grad:
            image_batch = image_batch.clone().detach().requires_grad_(True)

        x = self.backbone.conv1(image_batch)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        feature_map = x
        if require_feature_grad:
            feature_map.retain_grad()

        x = self.backbone.avgpool(x)
        embeddings = torch.flatten(x, 1)
        return feature_map, embeddings

    def forward(
        self,
        image_batch: torch.Tensor,
        require_feature_grad: bool = False,
    ) -> ForwardBatch:
        """Run the model once."""
        feature_map, embeddings = self.extract_features(
            image_batch,
            require_feature_grad=require_feature_grad,
        )
        logits = self.head(embeddings)
        return ForwardBatch(
            logits=logits,
            embeddings=embeddings,
            feature_map=feature_map,
        )
