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

from src.models import build_model
from src.dataloader import create_dataloaders


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested_device)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path} does not contain model_state_dict")

    return checkpoint


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


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict[str, float]:
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

    try:
        mean_ap = average_precision_score(
            targets,
            probs,
            average="macro",
        )
    except ValueError:
        mean_ap = float("nan")

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "map": float(mean_ap),
    }


def idx_to_genre_mapping(genre_to_idx: dict[str, int]) -> dict[int, str]:
    return {idx: genre for genre, idx in genre_to_idx.items()}


def top_k_genres(
    probs: np.ndarray,
    sample_index: int,
    idx_to_genre: dict[int, str],
    top_k: int,
) -> list[str]:
    scores = probs[sample_index]
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        f"{idx_to_genre[idx]} ({scores[idx]:.2f})"
        for idx in top_indices
    ]


def unnormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized CHW tensor to a displayable HWC numpy image.
    Assumes ImageNet normalization, which is what torchvision/timm/CLIP-style
    image pipelines commonly use in this project.
    """
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    return image


def collect_illustration_batch(test_loader, num_examples: int):
    images = []
    titles = []
    targets = []

    for batch in test_loader:
        batch_images = batch["image"]
        batch_targets = batch["target"]
        batch_titles = batch.get("title", None)

        for i in range(batch_images.shape[0]):
            images.append(batch_images[i])
            targets.append(batch_targets[i].detach().cpu().numpy())

            if batch_titles is not None:
                titles.append(str(batch_titles[i]))
            else:
                titles.append(f"Example {len(images)}")

            if len(images) >= num_examples:
                return images, titles, targets

    return images, titles, targets

def true_genres(
    target: np.ndarray,
    idx_to_genre: dict[int, str],
) -> list[str]:
    indices = np.where(target > 0.5)[0]

    return [
        idx_to_genre[idx]
        for idx in indices
    ]

def plot_prediction_grid(
    illustration_images: list[torch.Tensor],
    illustration_titles: list[str],
    illustration_targets: list[np.ndarray],
    model_probs: dict[str, np.ndarray],
    idx_to_genre: dict[int, str],
    top_k: int,
    output_path: str | Path,
) -> None:
    num_examples = len(illustration_images)
    model_names = list(model_probs.keys())

    row_labels = ["Poster", "True"] + model_names
    num_rows = len(row_labels)
    num_cols = num_examples

    fig_width = max(3.0 * num_cols, 8.0)
    fig_height = 2.6 + 1.0 * (num_rows - 1)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        gridspec_kw={
            "height_ratios": [2.2] + [1.0] * (num_rows - 1),
            "hspace": 0.35,
            "wspace": 0.25,
        },
    )

    # Poster row
    for col in range(num_cols):
        ax = axes[0, col]
        ax.imshow(unnormalize_image(illustration_images[col]))
        ax.set_title(illustration_titles[col][:35], fontsize=8, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # True-label row
    for col in range(num_cols):
        ax = axes[1, col]
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        labels = true_genres(
            target=illustration_targets[col],
            idx_to_genre=idx_to_genre,
        )

        text = "\n".join(labels)

        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=8,
            transform=ax.transAxes,
            wrap=True,
        )

    # Model prediction rows
    for model_row, model_name in enumerate(model_names, start=2):
        probs = model_probs[model_name]

        for col in range(num_cols):
            ax = axes[model_row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            genres = top_k_genres(
                probs=probs,
                sample_index=col,
                idx_to_genre=idx_to_genre,
                top_k=top_k,
            )

            text = "\n".join(genres)

            ax.text(
                0.5,
                0.5,
                text,
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
                wrap=True,
            )

    # Add row labels outside the grid, aligned vertically by row.
    for row_idx, row_label in enumerate(row_labels):
        ax = axes[row_idx, 0]
        ax.text(
            -0.18,
            0.5,
            row_label,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold" if row_idx <= 1 else "normal",
            transform=ax.transAxes,
        )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved qualitative prediction grid to {output_path}")
def evaluate_model_spec(
    label: str,
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    threshold: float,
):
    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path, device=device)

    test_loader, genre_to_idx = build_test_loader(config)

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

    metrics = compute_metrics(
        probs=probs,
        targets=targets,
        threshold=threshold,
    )

    row = {
        "model": label,
        "checkpoint": checkpoint_path,
        "threshold": threshold,
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "map": metrics["map"],
    }

    return row, probs, targets, test_loader, genre_to_idx


def parse_model_specs(model_args: list[list[str]]) -> list[tuple[str, str, str]]:
    specs = []

    for item in model_args:
        if len(item) != 3:
            raise ValueError(
                "Each --model entry must contain exactly three values: "
                "LABEL CONFIG_PATH CHECKPOINT_PATH"
            )

        label, config_path, checkpoint_path = item
        specs.append((label, config_path, checkpoint_path))

    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more trained multilabel poster classifiers on the test set "
            "and optionally create a qualitative prediction grid."
        )
    )

    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "CONFIG", "CHECKPOINT"),
        help=(
            "Model specification. Use once per model: "
            "--model ResNet configs/resnet.yaml checkpoints/resnet/best.pt"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for micro-F1 and macro-F1.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cuda, or cpu.",
    )

    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to save the evaluation table as CSV.",
    )

    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional path to save qualitative prediction grid.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of test posters to show in the qualitative grid.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predicted genres to show per model and poster.",
    )

    args = parser.parse_args()

    device = resolve_device(args.device)
    model_specs = parse_model_specs(args.model)

    rows = []
    all_probs = {}

    reference_targets = None
    reference_test_loader = None
    reference_genre_to_idx = None

    for label, config_path, checkpoint_path in model_specs:
        row, probs, targets, test_loader, genre_to_idx = evaluate_model_spec(
            label=label,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=args.threshold,
        )

        rows.append(row)
        all_probs[label] = probs

        if reference_targets is None:
            reference_targets = targets
            reference_test_loader = test_loader
            reference_genre_to_idx = genre_to_idx
        else:
            if targets.shape != reference_targets.shape:
                raise ValueError(
                    f"Target shape mismatch for model {label}: "
                    f"got {targets.shape}, expected {reference_targets.shape}."
                )

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("map", ascending=False)

    print("\nTest-set evaluation:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.csv_output is not None:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_output, index=False)
        print(f"\nSaved evaluation table to {csv_output}")

    if args.plot_output is not None:
        if reference_test_loader is None or reference_genre_to_idx is None:
            raise RuntimeError("No reference test loader available for plotting.")

        illustration_images, illustration_titles, illustration_targets = collect_illustration_batch(
            test_loader=reference_test_loader,
            num_examples=args.num_examples,
        )
        
        idx_to_genre = idx_to_genre_mapping(reference_genre_to_idx)
        
        plot_prediction_grid(
            illustration_images=illustration_images,
            illustration_titles=illustration_titles,
            illustration_targets=illustration_targets,
            model_probs=all_probs,
            idx_to_genre=idx_to_genre,
            top_k=args.top_k,
            output_path=args.plot_output,
        )

if __name__ == "__main__":
    main()
