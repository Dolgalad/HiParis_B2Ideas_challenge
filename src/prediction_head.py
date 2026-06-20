from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """
    Configurable multilayer prediction head for multilabel classification.

    The output is raw logits. Do not apply sigmoid here.
    BCEWithLogitsLoss expects logits directly.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_hidden_layers: int = 1,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()

        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")

        if activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "gelu":
            activation_layer = nn.GELU
        elif activation == "silu":
            activation_layer = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        current_dim = in_features

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(activation_layer())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def build_prediction_head(
    in_features: int,
    num_classes: int,
    head_config: dict | None = None,
) -> nn.Module:
    """
    Build a prediction head from config.

    Backward compatible default:
    If no head_config is provided, this returns a simple Dropout -> Linear head,
    matching the previous behavior.
    """
    head_config = head_config or {}

    head_type = head_config.get("type", "linear")
    dropout = head_config.get("dropout", 0.2)

    if head_type == "linear":
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    if head_type == "mlp":
        return MLPHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dim=head_config.get("hidden_dim", 512),
            num_hidden_layers=head_config.get("num_hidden_layers", 1),
            dropout=dropout,
            use_layer_norm=head_config.get("use_layer_norm", True),
            activation=head_config.get("activation", "gelu"),
        )

    raise ValueError(f"Unknown prediction head type: {head_type}")
