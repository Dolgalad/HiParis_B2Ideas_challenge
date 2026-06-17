from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision import models


#class ResNetMultiLabelClassifier(nn.Module):
#    def __init__(
#        self,
#        num_classes: int,
#        backbone_name: str = "resnet50",
#        pretrained: bool = True,
#        dropout: float = 0.2,
#    ):
#        super().__init__()
#
#        if backbone_name != "resnet50":
#            raise ValueError(f"Unsupported ResNet backbone: {backbone_name}")
#
#        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
#        self.backbone = models.resnet50(weights=weights)
#
#        in_features = self.backbone.fc.in_features
#        self.backbone.fc = nn.Sequential(
#            nn.Dropout(dropout),
#            nn.Linear(in_features, num_classes),
#        )
#
#    def forward(self, images: torch.Tensor) -> torch.Tensor:
#        return self.backbone(images)

class ResNetMultiLabelClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if backbone_name != "resnet50":
            raise ValueError(f"Unsupported ResNet backbone: {backbone_name}")

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # Make sure the classification head is trainable.
        for param in self.backbone.fc.parameters():
            param.requires_grad = True


class CLIPPosterClassifier(nn.Module):
    """
    CLIP image encoder + multi-label genre classification head.

    This uses CLIP's visual encoder only. It should be useful for posters because
    CLIP has strong image-text pretraining, but this is still not the same as OCR.
    """

    def __init__(
        self,
        num_classes: int,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open-clip-torch is required for the CLIP model. "
                "Install it with: pip install open-clip-torch"
            ) from exc

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained,
        )

        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        feature_dim = self.clip_model.visual.output_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(images)
        features = features.float()
        logits = self.classifier(features)
        return logits


def build_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    model_config = config["model"]
    model_name = model_config["name"]

    #if model_name == "resnet50":
    #    return ResNetMultiLabelClassifier(
    #        num_classes=num_classes,
    #        backbone_name="resnet50",
    #        pretrained=model_config.get("pretrained", True),
    #        dropout=model_config.get("dropout", 0.2),
    #    )

    if model_name == "resnet50":
        return ResNetMultiLabelClassifier(
            num_classes=num_classes,
            backbone_name="resnet50",
            pretrained=model_config.get("pretrained", True),
            dropout=model_config.get("dropout", 0.2),
            freeze_backbone=model_config.get("freeze_backbone", False),
        )

    if model_name == "clip":
        return CLIPPosterClassifier(
            num_classes=num_classes,
            clip_model_name=model_config.get("clip_model_name", "ViT-B-32"),
            clip_pretrained=model_config.get("clip_pretrained", "openai"),
            freeze_backbone=model_config.get("freeze_backbone", False),
            dropout=model_config.get("dropout", 0.2),
        )

    raise ValueError(f"Unknown model name: {model_name}")
