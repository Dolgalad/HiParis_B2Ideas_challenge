#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score


# ---------------------------------------------------------------------
# Adjust these imports if your project uses different function names.
# Use the same functions that scripts/train.py uses.
# ---------------------------------------------------------------------
from src.models import build_model
from src.dataloader import create_dataloaders
# ---------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested_device)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required_keys = {"model_state_dict"}
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise ValueError(
            f"Checkpoint is missing required keys: {sorted(missing)}"
        )

    return checkpoint


def build_validation_loader(config: dict[str, Any]):
    """
    Rebuild dataloaders from the config and return the validation loader.

    create_dataloaders expects explicit keyword arguments, not the full config dict.
    """
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    dataloader_output = create_dataloaders(
        data_dir=data_config.get("data_dir", "data"),
        image_size=data_config.get("image_size", 224),
        batch_size=training_config.get("batch_size", 32),
        num_workers=data_config.get("num_workers", 4),
        use_cache=data_config.get("use_cache", True),
    )

    if not isinstance(dataloader_output, tuple) or len(dataloader_output) != 4:
        raise ValueError(
            "Expected create_dataloaders(...) to return "
            "(train_loader, val_loader, test_loader, genre_to_idx)."
        )

    train_loader, val_loader, test_loader, genre_to_idx = dataloader_output

    return val_loader, genre_to_idx

def build_model_from_config(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    device: torch.device,
    genre_to_idx: dict[str, int] | None = None,
):
    """
    Build model from config and load checkpoint weights.
    """
    checkpoint_genre_to_idx = checkpoint.get("genre_to_idx", None)

    if checkpoint_genre_to_idx is not None:
        num_classes = len(checkpoint_genre_to_idx)
    elif genre_to_idx is not None:
        num_classes = len(genre_to_idx)
    else:
        num_classes = config["model"].get("num_classes", None)

    if num_classes is None:
        raise ValueError(
            "Could not infer num_classes. Expected checkpoint['genre_to_idx'], "
            "dataloader genre_to_idx, or config['model']['num_classes']."
        )

    model = build_model(config, num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model

@torch.no_grad()
def compute_validation_probabilities(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_probs = []
    all_targets = []

    for batch in tqdm(val_loader, desc="Computing validation probabilities"):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    return probs, targets


def compute_threshold_sweep(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)

        micro_f1 = f1_score(
            targets,
            preds,
            average="micro",
            zero_division=0,
        )

        macro_f1 = f1_score(
            targets,
            preds,
            average="macro",
            zero_division=0,
        )

        rows.append(
            {
                "threshold": float(threshold),
                "micro_f1": float(micro_f1),
                "macro_f1": float(macro_f1),
            }
        )

    return pd.DataFrame(rows)


def add_non_dominated_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark non-dominated thresholds for maximizing both micro_f1 and macro_f1.
    """
    values = df[["micro_f1", "macro_f1"]].to_numpy()
    is_non_dominated = []

    for i, point in enumerate(values):
        other = np.delete(values, i, axis=0)

        dominated = np.any(
            (other[:, 0] >= point[0])
            & (other[:, 1] >= point[1])
            & (
                (other[:, 0] > point[0])
                | (other[:, 1] > point[1])
            )
        )

        is_non_dominated.append(not dominated)

    df = df.copy()
    df["is_non_dominated"] = is_non_dominated

    return df


def plot_threshold_tradeoff(
    df: pd.DataFrame,
    title: str,
    output_path: str | Path | None,
) -> None:
    frontier = df[df["is_non_dominated"]].sort_values("micro_f1")

    plt.figure(figsize=(7, 6))

    plt.scatter(
        df["micro_f1"],
        df["macro_f1"],
        alpha=0.45,
        label="Thresholds",
    )

    plt.plot(
        frontier["micro_f1"],
        frontier["macro_f1"],
        marker="o",
        linewidth=2,
        label="Non-dominated frontier",
    )

    best_micro = df.loc[df["micro_f1"].idxmax()]
    best_macro = df.loc[df["macro_f1"].idxmax()]

    plt.scatter(
        [best_micro["micro_f1"]],
        [best_micro["macro_f1"]],
        marker="x",
        s=100,
        label=f"Best micro-F1, t={best_micro['threshold']:.3f}",
    )

    plt.scatter(
        [best_macro["micro_f1"]],
        [best_macro["macro_f1"]],
        marker="x",
        s=100,
        label=f"Best macro-F1, t={best_macro['threshold']:.3f}",
    )

    for _, row in frontier.iterrows():
        plt.annotate(
            f"{row['threshold']:.2f}",
            (row["micro_f1"], row["macro_f1"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    plt.title(title)
    plt.xlabel("Micro-F1")
    plt.ylabel("Macro-F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def print_summary(df: pd.DataFrame, probs: np.ndarray, targets: np.ndarray) -> None:
    val_map = average_precision_score(
        targets,
        probs,
        average="macro",
    )

    best_micro = df.loc[df["micro_f1"].idxmax()]
    best_macro = df.loc[df["macro_f1"].idxmax()]
    frontier = df[df["is_non_dominated"]].sort_values("micro_f1")

    print("\nValidation mAP:")
    print(f"{val_map:.6f}")

    print("\nBest threshold by micro-F1:")
    print(best_micro.to_string())

    print("\nBest threshold by macro-F1:")
    print(best_macro.to_string())

    print("\nNon-dominated thresholds:")
    print(frontier.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained model checkpoint, compute validation probabilities, "
            "sweep thresholds, and plot micro-F1 vs macro-F1."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to model config YAML file.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint, e.g. checkpoints/clip/best.pt.",
    )

    parser.add_argument(
        "--title",
        required=True,
        help="Plot title.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the plot.",
    )

    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to save threshold sweep results as CSV.",
    )

    parser.add_argument(
        "--probs-output",
        default=None,
        help="Optional path to save validation probabilities as .npy.",
    )

    parser.add_argument(
        "--targets-output",
        default=None,
        help="Optional path to save validation targets as .npy.",
    )

    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=101,
        help="Number of thresholds between 0 and 1.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cuda, or cpu.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, device=device)

    val_loader, dataloader_genre_to_idx = build_validation_loader(config)

    model = build_model_from_config(
        config=config,
        checkpoint=checkpoint,
        device=device,
        genre_to_idx=dataloader_genre_to_idx,
    ) 
    probs, targets = compute_validation_probabilities(
        model=model,
        val_loader=val_loader,
        device=device,
    )

    if args.probs_output is not None:
        probs_output = Path(args.probs_output)
        probs_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(probs_output, probs)
        print(f"Saved probabilities to {probs_output}")

    if args.targets_output is not None:
        targets_output = Path(args.targets_output)
        targets_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(targets_output, targets)
        print(f"Saved targets to {targets_output}")

    thresholds = np.linspace(
        0.0,
        1.0,
        args.num_thresholds,
    )

    df = compute_threshold_sweep(
        probs=probs,
        targets=targets,
        thresholds=thresholds,
    )

    df = add_non_dominated_flag(df)

    print_summary(
        df=df,
        probs=probs,
        targets=targets,
    )

    if args.csv_output is not None:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_output, index=False)
        print(f"Saved threshold sweep to {csv_output}")

    plot_threshold_tradeoff(
        df=df,
        title=args.title,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
