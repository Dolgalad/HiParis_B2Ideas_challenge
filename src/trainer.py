from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm

import csv
from torch.utils.tensorboard import SummaryWriter


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

        use_pos_weight = training_config.get("use_pos_weight", False)

        self.stages = self._get_training_stages()
        self.optimizer = None
        self.scheduler = None
        self.current_stage = None

        self.output_dir = Path(training_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.output_dir / "history.csv"
        self.writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")
        self.global_step = 0

        if use_pos_weight:
            pos_weight = self._compute_pos_weight()
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            for genre, idx in self.genre_to_idx.items():
                self.writer.add_scalar(f"ClassWeight/{genre}", pos_weight[idx].item(), 0)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Track best validation metrics per stage and globally
        # Track best validation metric overall and per stage
        if self.config["training"].get("save_best_metric_sense", "min") == "min":
            self.best_val_metric = float("inf")
            self.best_stage_metrics = {
                stage["index"]: float("inf") for stage in self.stages
            }
        else:
            self.best_val_metric = -float("inf")
            self.best_stage_metrics = {
                stage["index"]: -float("inf") for stage in self.stages
            }
    def _stage_checkpoint_slug(
        self,
        stage_index: int,
        stage_name: str,
    ) -> str:
        safe_stage_name = "".join(
            char if char.isalnum() or char in ["_", "-"] else "_"
            for char in stage_name
        )
    
        return f"stage_{stage_index}_{safe_stage_name}"

    def _compute_pos_weight(self) -> torch.Tensor:
        num_classes = len(self.genre_to_idx)
        positive_counts = torch.zeros(num_classes, dtype=torch.float32)
    
        dataset = self.train_loader.dataset
    
        for idx in range(len(dataset)):
            target = dataset.encode_genres(
                dataset.df.iloc[idx][dataset.genre_column]
            )
            positive_counts += target
    
        num_samples = len(dataset)
        negative_counts = num_samples - positive_counts
    
        pos_weight = negative_counts / positive_counts.clamp(min=1.0)
    
        max_pos_weight = self.config["training"].get("max_pos_weight", None)
        if max_pos_weight is not None:
            pos_weight = pos_weight.clamp(max=float(max_pos_weight))
    
        return pos_weight.to(self.device)
    def _get_training_stages(self) -> list[dict[str, Any]]:
        """
        Return a normalized list of training stages.
    
        Backward compatible behavior:
        If config["training"]["stages"] is missing, we create a single stage
        from the old keys:
            training.epochs
            training.lr
            training.classifier_lr
            training.scheduler
            model.freeze_backbone
        """
        training_config = self.config["training"]
        model_config = self.config["model"]
    
        default_classifier_patterns = training_config.get(
            "classifier_patterns",
            ["classifier", "head", "fc"],
        )
    
        if "stages" not in training_config:
            stage = {
                "index": 1,
                "name": "single_stage",
                "epochs": training_config["epochs"],
                "freeze_backbone": model_config.get("freeze_backbone", False),
                "lr": training_config["lr"],
                "classifier_lr": training_config.get("classifier_lr", None),
                "weight_decay": training_config.get("weight_decay", 0.0),
                "scheduler": training_config.get("scheduler", "none"),
                "min_lr": training_config.get("min_lr", 0.0),
                "classifier_patterns": default_classifier_patterns,
            }
            return [stage]
    
        stages = []
    
        for idx, stage_config in enumerate(training_config["stages"], start=1):
            stage = {
                "index": idx,
                "name": stage_config.get("name", f"stage_{idx}"),
                "epochs": stage_config["epochs"],
                "freeze_backbone": stage_config.get("freeze_backbone", False),
                "lr": stage_config.get("lr", training_config.get("lr", 1e-4)),
                "classifier_lr": stage_config.get(
                    "classifier_lr",
                    training_config.get("classifier_lr", None),
                ),
                "weight_decay": stage_config.get(
                    "weight_decay",
                    training_config.get("weight_decay", 0.0),
                ),
                "scheduler": stage_config.get(
                    "scheduler",
                    training_config.get("scheduler", "none"),
                ),
                "min_lr": stage_config.get(
                    "min_lr",
                    training_config.get("min_lr", 0.0),
                ),
                "classifier_patterns": stage_config.get(
                    "classifier_patterns",
                    default_classifier_patterns,
                ),
            }
    
            stages.append(stage)

        return stages


    def _is_classifier_parameter(
        self,
        parameter_name: str,
        classifier_patterns: list[str],
    ) -> bool:
        return any(pattern in parameter_name for pattern in classifier_patterns)
    
    
    def _set_trainable_for_stage(self, stage: dict[str, Any]) -> None:
        """
        Freeze or unfreeze parameters for the current stage.
    
        If freeze_backbone=True:
            - all params are frozen
            - classifier/head/fc params are unfrozen
    
        If freeze_backbone=False:
            - all params are trainable
        """
        freeze_backbone = stage.get("freeze_backbone", False)
        classifier_patterns = stage.get("classifier_patterns", ["classifier", "head", "fc"])
    
        if not freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = True
            return
    
        for param in self.model.parameters():
            param.requires_grad = False
    
        num_trainable = 0
    
        for name, param in self.model.named_parameters():
            if self._is_classifier_parameter(name, classifier_patterns):
                param.requires_grad = True
                num_trainable += param.numel()
    
        if num_trainable == 0:
            raise RuntimeError(
                "freeze_backbone=True, but no classifier parameters were found. "
                f"Checked patterns: {classifier_patterns}. "
                "Add the correct classifier parameter name to classifier_patterns."
            )
    
    
    def _count_trainable_parameters(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def _build_optimizer(self, stage: dict[str, Any]):
        lr = stage["lr"]
        classifier_lr = stage.get("classifier_lr", None)
        weight_decay = stage.get("weight_decay", 0.0)
        classifier_patterns = stage.get("classifier_patterns", ["classifier", "head", "fc"])
    
        if classifier_lr is not None:
            classifier_params = []
            backbone_params = []
    
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
    
                if self._is_classifier_parameter(name, classifier_patterns):
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
    
            param_groups = []
    
            if len(backbone_params) > 0:
                param_groups.append(
                    {
                        "params": backbone_params,
                        "lr": lr,
                        "name": "backbone",
                    }
                )
    
            if len(classifier_params) > 0:
                param_groups.append(
                    {
                        "params": classifier_params,
                        "lr": classifier_lr,
                        "name": "classifier",
                    }
                )
    
            if len(param_groups) == 0:
                raise RuntimeError("No trainable parameters found for optimizer.")
    
            return torch.optim.AdamW(
                param_groups,
                weight_decay=weight_decay,
            )
    
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
    
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found for optimizer.")
    
        return torch.optim.AdamW(
            [
                {
                    "params": trainable_params,
                    "lr": lr,
                    "name": "model",
                }
            ],
            weight_decay=weight_decay,
        )

    def _build_scheduler(self, stage: dict[str, Any]):
        scheduler_name = stage.get("scheduler", "none")
    
        if scheduler_name is None or scheduler_name == "none":
            return None
    
        if scheduler_name == "cosine":
            epochs = stage["epochs"]
            min_lr = stage.get("min_lr", 0.0)
    
            total_steps = epochs * len(self.train_loader)
    
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=min_lr,
            )
    
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    def train_one_epoch(
            self, 
            epoch: int,
            stage_index: int, 
            stage_name: str,
            stage_epoch,
    ) -> dict[str, float]:
        self.model.train()

        total_loss = 0.0
        all_logits = []
        all_targets = []

        progress = tqdm(
            self.train_loader,
            desc=f"Train epoch {epoch} | stage {stage_index}:{stage_name} [{stage_epoch}]",
            leave=False,
        )

        for batch in progress:
            images, targets = move_batch_to_device(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1

            log_every_n_steps = self.config["training"].get("log_every_n_steps", 20)
            if self.global_step % log_every_n_steps == 0:
                self.writer.add_scalar("Loss/train_step", loss.item(), self.global_step)
                self.writer.add_scalar("Stage/index", stage_index, self.global_step)
                self.writer.add_scalar("Stage/epoch", stage_epoch, self.global_step)
            
                for idx, param_group in enumerate(self.optimizer.param_groups):
                    group_name = param_group.get("name", f"group_{idx}")
                    self.writer.add_scalar(
                        f"LR/{group_name}",
                        param_group["lr"],
                        self.global_step,
                    )

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
    def validate(
            self, 
            epoch: int,
            stage_index: int,
            stage_name: str,
            stage_epoch: int,
    ) -> dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        all_logits = []
        all_targets = []

        progress = tqdm(
            self.val_loader,
            desc=f"Val epoch {epoch} | stage {stage_index}:{stage_name} [{stage_epoch}]",
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
        stage_index: int,
        stage_name: str,
        stage_epoch: int,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "stage_index": stage_index,
            "stage_name": stage_name,
            "stage_epoch": stage_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "config": self.config,
            "genre_to_idx": self.genre_to_idx,
            "val_metrics": val_metrics,
        }

        torch.save(checkpoint, path)

    def fit(self) -> None:
        global_epoch = 0
    
        save_best_metric = self.config["training"].get("save_best_metric", "loss")
        save_best_metric_sense = self.config["training"].get(
            "save_best_metric_sense",
            "min",
        )
    
        total_epochs = sum(stage["epochs"] for stage in self.stages)
    
        for stage in self.stages:
            stage_index = stage["index"]
            stage_name = stage["name"]
            stage_epochs = stage["epochs"]
    
            self.current_stage = stage
    
            self._set_trainable_for_stage(stage)
            trainable_params, total_params = self._count_trainable_parameters()
    
            self.optimizer = self._build_optimizer(stage)
            self.scheduler = self._build_scheduler(stage)
    
            print(
                f"\nStarting stage {stage_index}: {stage_name} | "
                f"epochs={stage_epochs} | "
                f"freeze_backbone={stage.get('freeze_backbone', False)} | "
                f"trainable_params={trainable_params:,}/{total_params:,}"
            )
    
            for stage_epoch in range(1, stage_epochs + 1):
                global_epoch += 1
    
                train_metrics = self.train_one_epoch(
                    epoch=global_epoch,
                    stage_index=stage_index,
                    stage_name=stage_name,
                    stage_epoch=stage_epoch,
                )
    
                val_metrics = self.validate(
                    epoch=global_epoch,
                    stage_index=stage_index,
                    stage_name=stage_name,
                    stage_epoch=stage_epoch,
                )
    
                print(
                    f"Epoch {global_epoch:03d}/{total_epochs:03d} | "
                    f"stage={stage_index}:{stage_name} "
                    f"stage_epoch={stage_epoch:03d}/{stage_epochs:03d} | "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"train_micro_f1={train_metrics['micro_f1']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_micro_f1={val_metrics['micro_f1']:.4f} "
                    f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                    f"val_map={val_metrics['map']:.4f}"
                )
    
                self.writer.add_scalar("Loss/train_epoch", train_metrics["loss"], global_epoch)
                self.writer.add_scalar("Loss/val_epoch", val_metrics["loss"], global_epoch)
    
                self.writer.add_scalar("Metrics/train_micro_f1", train_metrics["micro_f1"], global_epoch)
                self.writer.add_scalar("Metrics/train_macro_f1", train_metrics["macro_f1"], global_epoch)
                self.writer.add_scalar("Metrics/train_map", train_metrics["map"], global_epoch)
    
                self.writer.add_scalar("Metrics/val_micro_f1", val_metrics["micro_f1"], global_epoch)
                self.writer.add_scalar("Metrics/val_macro_f1", val_metrics["macro_f1"], global_epoch)
                self.writer.add_scalar("Metrics/val_map", val_metrics["map"], global_epoch)
    
                self.writer.add_scalar("Stage/index_epoch", stage_index, global_epoch)
                self.writer.add_scalar("Stage/stage_epoch", stage_epoch, global_epoch)
    
                checkpoint_metric = val_metrics[save_best_metric]
    
                is_best = (
                    checkpoint_metric < self.best_val_metric
                    if save_best_metric_sense == "min"
                    else checkpoint_metric > self.best_val_metric
                )

                stage_slug = self._stage_checkpoint_slug(stage_index, stage_name)

                # Checkpoint save: best model per stage is saved as well as global best
                latest_path = self.output_dir / "latest.pt"
                latest_stage_path = self.output_dir / f"latest_{stage_slug}.pt"
                
                self.save_checkpoint(
                    latest_path,
                    global_epoch,
                    val_metrics,
                    stage_index=stage_index,
                    stage_name=stage_name,
                    stage_epoch=stage_epoch,
                )
                
                self.save_checkpoint(
                    latest_stage_path,
                    global_epoch,
                    val_metrics,
                    stage_index=stage_index,
                    stage_name=stage_name,
                    stage_epoch=stage_epoch,
                )
                
                is_stage_best = (
                    checkpoint_metric < self.best_stage_metrics[stage_index]
                    if save_best_metric_sense == "min"
                    else checkpoint_metric > self.best_stage_metrics[stage_index]
                )
                
                if is_stage_best:
                    self.best_stage_metrics[stage_index] = checkpoint_metric
                    best_stage_path = self.output_dir / f"best_{stage_slug}.pt"
                
                    self.save_checkpoint(
                        best_stage_path,
                        global_epoch,
                        val_metrics,
                        stage_index=stage_index,
                        stage_name=stage_name,
                        stage_epoch=stage_epoch,
                    )
                
                    print(f"Saved new best checkpoint for stage {stage_index}:{stage_name} to {best_stage_path}")
                
                if is_best:
                    self.best_val_metric = checkpoint_metric
                    best_path = self.output_dir / "best.pt"
                
                    self.save_checkpoint(
                        best_path,
                        global_epoch,
                        val_metrics,
                        stage_index=stage_index,
                        stage_name=stage_name,
                        stage_epoch=stage_epoch,
                    )
                
                    print(f"Saved new overall best checkpoint to {best_path}")
    
    
                self._append_history_row(
                    epoch=global_epoch,
                    stage_index=stage_index,
                    stage_name=stage_name,
                    stage_epoch=stage_epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    checkpoint_metric=checkpoint_metric,
                    is_best=is_best,
                    is_stage_best=is_stage_best
                )
    
        self.writer.close()
    def _get_lrs(self) -> dict[str, float]:
        lrs = {}
    
        for idx, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get("name", f"group_{idx}")
            lrs[f"lr_{group_name}"] = param_group["lr"]
    
        return lrs
    
    
    def _append_history_row(
        self,
        epoch: int,
        stage_index: int,
        stage_name: str,
        stage_epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        checkpoint_metric: float,
        is_best: bool,
        is_stage_best: bool,
    ) -> None:
        row = {
            "epoch": epoch,
            "stage_index": stage_index,
            "stage_name": stage_name,
            "stage_epoch": stage_epoch,
            "train_loss": train_metrics["loss"],
            "train_micro_f1": train_metrics["micro_f1"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_map": train_metrics["map"],
            "val_loss": val_metrics["loss"],
            "val_micro_f1": val_metrics["micro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_map": val_metrics["map"],
            "checkpoint_metric": checkpoint_metric,
            "is_best": int(is_best),
            "is_stage_best": int(is_stage_best),
            **self._get_lrs(),
        }
    
        write_header = not self.history_path.exists()
    
        with open(self.history_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
    
            if write_header:
                writer.writeheader()
    
            writer.writerow(row)
