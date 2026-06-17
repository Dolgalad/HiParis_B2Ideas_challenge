from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm


def move_batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    return images, targets


@torch.no_grad()
def compute_multilabel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    y_pred = (probs >= threshold).astype(int)

    metrics = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        metrics["map"] = average_precision_score(y_true, probs, average="macro")
    except ValueError:
        metrics["map"] = float("nan")

    return metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict[str, Any],
        genre_to_idx: dict[str, int],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.genre_to_idx = genre_to_idx

        training_config = config["training"]

        requested_device = training_config.get("device", "auto")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(requested_device)
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = self._build_optimizer()

        self.output_dir = Path(training_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")

    def _build_optimizer(self):
        training_config = self.config["training"]
        model_config = self.config["model"]

        lr = training_config["lr"]
        weight_decay = training_config.get("weight_decay", 0.0)

        if model_config["name"] == "clip" and "classifier_lr" in training_config:
            classifier_lr = training_config["classifier_lr"]

            classifier_params = []
            backbone_params = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if "classifier" in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)

            return torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": lr},
                    {"params": classifier_params, "lr": classifier_lr},
                ],
                weight_decay=weight_decay,
            )

        return torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()

        total_loss = 0.0
        all_logits = []
        all_targets = []

        progress = tqdm(
            self.train_loader,
            desc=f"Train epoch {epoch}",
            leave=False,
        )

        for batch in progress:
            images, targets = move_batch_to_device(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

            all_logits.append(logits.detach())
            all_targets.append(targets.detach())

            progress.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(self.train_loader.dataset)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_multilabel_metrics(all_logits, all_targets)

        return {
            "loss": epoch_loss,
            **metrics,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        all_logits = []
        all_targets = []

        progress = tqdm(
            self.val_loader,
            desc=f"Val epoch {epoch}",
            leave=False,
        )

        for batch in progress:
            images, targets = move_batch_to_device(batch, self.device)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            total_loss += loss.item() * images.size(0)

            all_logits.append(logits)
            all_targets.append(targets)

            progress.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(self.val_loader.dataset)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_multilabel_metrics(all_logits, all_targets)

        return {
            "loss": epoch_loss,
            **metrics,
        }

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_metrics: dict[str, float],
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "genre_to_idx": self.genre_to_idx,
            "val_metrics": val_metrics,
        }

        torch.save(checkpoint, path)

    def fit(self) -> None:
        epochs = self.config["training"]["epochs"]

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_micro_f1={train_metrics['micro_f1']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_micro_f1={val_metrics['micro_f1']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_map={val_metrics['map']:.4f}"
            )

            latest_path = self.output_dir / "latest.pt"
            self.save_checkpoint(latest_path, epoch, val_metrics)

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                best_path = self.output_dir / "best.pt"
                self.save_checkpoint(best_path, epoch, val_metrics)
                print(f"Saved new best checkpoint to {best_path}")
