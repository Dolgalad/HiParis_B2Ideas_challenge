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
from sklearn.metrics import f1_score

from src.models import build_model
from src.dataloader import create_dataloaders


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested_device)


def get_data_config(config: dict[str, Any]) -> tuple[str, int, int, int, bool]:
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    data_dir = data_config.get("data_dir", "data")
    image_size = data_config.get("image_size", 224)
    batch_size = training_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)
    use_cache = data_config.get("use_cache", True)

    return data_dir, image_size, batch_size, num_workers, use_cache


def build_test_loader(config: dict[str, Any]):
    data_dir, image_size, batch_size, num_workers, use_cache = get_data_config(config)

    train_loader, val_loader, test_loader, genre_to_idx = create_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        use_cache=use_cache,
    )

    return test_loader, genre_to_idx


def find_best_checkpoint(config: dict[str, Any]) -> Path:
    training_config = config.get("training", {})
    output_dir = training_config.get("output_dir", None)

    if output_dir is None:
        raise ValueError("Missing training.output_dir in config.")

    checkpoint_path = Path(output_dir) / "best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path} does not contain model_state_dict")

    return checkpoint


def build_loaded_model(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    genre_to_idx: dict[str, int],
    device: torch.device,
) -> torch.nn.Module:
    checkpoint_genre_to_idx = checkpoint.get("genre_to_idx", None)

    if checkpoint_genre_to_idx is not None:
        num_classes = len(checkpoint_genre_to_idx)
    else:
        num_classes = len(genre_to_idx)

    model = build_model(config, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict_on_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    all_probs = []
    all_targets = []

    for batch in tqdm(loader, desc=desc):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    return probs, targets


def compute_per_class_f1(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> np.ndarray:
    preds = (probs >= threshold).astype(int)

    return f1_score(
        targets,
        preds,
        average=None,
        zero_division=0,
    )


def idx_to_genre_mapping(genre_to_idx: dict[str, int]) -> dict[int, str]:
    return {idx: genre for genre, idx in genre_to_idx.items()}


def evaluate_model(
    label: str,
    config_path: str | Path,
    device: torch.device,
    threshold: float,
    loader_cache: dict[str, tuple],
):
    config = load_config(config_path)
    checkpoint_path = find_best_checkpoint(config)

    cache_key = str(config_path)
    if cache_key not in loader_cache:
        loader_cache[cache_key] = build_test_loader(config)

    test_loader, genre_to_idx = loader_cache[cache_key]

    checkpoint = load_checkpoint(checkpoint_path, device=device)

    model = build_loaded_model(
        config=config,
        checkpoint=checkpoint,
        genre_to_idx=genre_to_idx,
        device=device,
    )

    probs, targets = predict_on_loader(
        model=model,
        loader=test_loader,
        device=device,
        desc=f"Evaluating {label}",
    )

    per_class_f1 = compute_per_class_f1(
        probs=probs,
        targets=targets,
        threshold=threshold,
    )

    idx_to_genre = idx_to_genre_mapping(genre_to_idx)
    genres = [idx_to_genre[i] for i in range(len(idx_to_genre))]

    return {
        "label": label,
        "checkpoint": checkpoint_path,
        "genres": genres,
        "per_class_f1": per_class_f1,
        "targets": targets,
    }


def make_results_table(model_results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for result in model_results:
        label = result["label"]
        checkpoint = result["checkpoint"]
        genres = result["genres"]
        per_class_f1 = result["per_class_f1"]
        targets = result["targets"]

        supports = targets.sum(axis=0).astype(int)

        for genre, f1, support in zip(genres, per_class_f1, supports):
            rows.append(
                {
                    "model": label,
                    "checkpoint": str(checkpoint),
                    "genre": genre,
                    "f1": float(f1),
                    "support": int(support),
                }
            )

    return pd.DataFrame(rows)


def plot_grouped_per_class_f1(
    results_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    figsize: tuple[float, float],
) -> None:
    pivot_df = results_df.pivot(
        index="genre",
        columns="model",
        values="f1",
    )

    # Keep genres in descending average F1. This usually makes the plot easier to read.
    pivot_df = pivot_df.loc[pivot_df.mean(axis=1).sort_values(ascending=False).index]

    genres = list(pivot_df.index)
    model_names = list(pivot_df.columns)

    x = np.arange(len(genres))
    num_models = len(model_names)
    bar_width = min(0.8 / max(num_models, 1), 0.25)

    fig, ax = plt.subplots(figsize=figsize)

    for model_idx, model_name in enumerate(model_names):
        offsets = (model_idx - (num_models - 1) / 2) * bar_width
        ax.bar(
            x + offsets,
            pivot_df[model_name].values,
            width=bar_width,
            label=model_name,
        )

    ax.set_ylabel("F1 score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(genres, rotation=45, ha="right")
    ax.legend(frameon=False, ncols=min(num_models, 3))

    if title:
        ax.set_title(title)

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved per-class F1 plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load best checkpoints for several multilabel poster classifiers, "
            "compute per-class F1 on the test set, and plot grouped bars by genre."
        )
    )

    parser.add_argument(
        "--model",
        nargs=2,
        action="append",
        required=True,
        metavar=("LABEL", "CONFIG_PATH"),
        help="Model specification: --model ResNet configs/resnet50_2stage.yaml",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used to binarize sigmoid probabilities.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cuda, or cpu.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where the grouped bar plot will be saved.",
    )

    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to save the per-class F1 table as CSV.",
    )

    parser.add_argument(
        "--title",
        default="Per-genre F1 score on the test set",
        help="Plot title. Use --title \"\" for no title.",
    )

    parser.add_argument(
        "--fig-width",
        type=float,
        default=7.0,
        help="Figure width in inches.",
    )

    parser.add_argument(
        "--fig-height",
        type=float,
        default=4.2,
        help="Figure height in inches.",
    )

    args = parser.parse_args()

    device = resolve_device(args.device)
    loader_cache = {}

    model_results = []

    for label, config_path in args.model:
        result = evaluate_model(
            label=label,
            config_path=config_path,
            device=device,
            threshold=args.threshold,
            loader_cache=loader_cache,
        )
        model_results.append(result)

    results_df = make_results_table(model_results)

    print("\nPer-class F1 scores:")
    print(
        results_df.pivot(
            index="genre",
            columns="model",
            values="f1",
        ).to_string(float_format=lambda x: f"{x:.4f}")
    )

    if args.csv_output is not None:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_output, index=False)
        print(f"\nSaved per-class F1 table to {csv_output}")

    plot_grouped_per_class_f1(
        results_df=results_df,
        output_path=args.output,
        title=args.title,
        figsize=(args.fig_width, args.fig_height),
    )


if __name__ == "__main__":
    main()
