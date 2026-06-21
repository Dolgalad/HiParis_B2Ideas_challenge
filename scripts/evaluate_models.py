#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import re

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

def draw_colored_word_list(
    ax,
    words,
    colors,
    x=0.5,
    y=0.5,
    fontsize=8,
    line_spacing=1.25,
    **kwargs,
):
    """
    Draw a vertically centered list of words, each with its own color.

    words: list[str]
    colors: list[str]
    x, y: position in axes coordinates
    line_spacing: multiplier relative to fontsize
    """

    assert len(words) == len(colors)

    n = len(words)
    spacing_pts = fontsize * line_spacing

    for i, (word, color) in enumerate(zip(words, colors)):
        # Center the whole block around y
        offset_y = ((n - 1) / 2 - i) * spacing_pts

        ax.annotate(
            word,
            xy=(x, y),
            xycoords=ax.transAxes,
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            **kwargs,
        )

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
    model_names = [n for n in model_names if "best_overall" in n]

    row_labels = ["", "Vrais genres"] + model_names
    num_rows = len(row_labels)
    num_cols = num_examples

    fig_width = max(1.7 * num_cols, 8.0)
    fig_height = 2.6 + 1.0 * (num_rows - 1)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        gridspec_kw={
            "height_ratios": [2.2] + [1.0] * (num_rows - 1),
            "hspace": 0.35,
            "wspace": 0.00,
        },
    )

    # Poster row
    for col in range(num_cols):
        ax = axes[0, col]
        ax.imshow(unnormalize_image(illustration_images[col]))
        #ax.set_title(illustration_titles[col][:35], fontsize=8, pad=8)
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
        if not "best_overall" in model_name:
            continue
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
            labels = true_genres(
                target=illustration_targets[col],
                idx_to_genre=idx_to_genre,
            )

            colors = ["g" if g.split("(")[0].strip() in labels else "r" for g in genres]
            draw_colored_word_list(
                ax,
                genres,
                colors,
                x=0.5,
                y=0.5,
                fontsize=8,
            )

    # Add row labels outside the grid, aligned vertically by row.
    for row_idx, row_label in enumerate(row_labels):
        if "best_overall" in row_label:
            row_label = row_label.replace("/best_overall", "")
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
    checkpoint_path: str | Path,
    device: torch.device,
    threshold: float,
    config: dict[str, Any] | None = None,
    test_loader=None,
    genre_to_idx: dict[str, int] | None = None,
):
    if config is None:
        config = load_config(config_path)

    checkpoint = load_checkpoint(checkpoint_path, device=device)

    if test_loader is None or genre_to_idx is None:
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
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch", ""),
        "checkpoint_stage": checkpoint.get("stage_name", ""),
        "checkpoint_stage_epoch": checkpoint.get("stage_epoch", ""),
        "val_loss": checkpoint.get("val_metrics", {}).get("loss", float("nan")),
        "val_micro_f1": checkpoint.get("val_metrics", {}).get("micro_f1", float("nan")),
        "val_macro_f1": checkpoint.get("val_metrics", {}).get("macro_f1", float("nan")),
        "val_map": checkpoint.get("val_metrics", {}).get("map", float("nan")),
        "threshold": threshold,
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "map": metrics["map"],
    }

    return row, probs, targets, test_loader, genre_to_idx


def stage_checkpoint_slug(stage_index: int, stage_name: str) -> str:
    safe_stage_name = "".join(
        char if char.isalnum() or char in ["_", "-"] else "_"
        for char in stage_name
    )

    return f"stage_{stage_index}_{safe_stage_name}"


def checkpoint_label(base_label: str, checkpoint_path: str | Path) -> str:
    checkpoint_path = Path(checkpoint_path)
    stem = checkpoint_path.stem

    if stem == "best":
        return f"{base_label}/best_overall"

    if stem.startswith("best_stage_"):
        stage_name = re.sub(r"^best_stage_\d+_", "", stem)
        return f"{base_label}/{stage_name}"

    return f"{base_label}/{stem}"


def discover_checkpoints_from_config(
    config: dict[str, Any],
    include_overall_best: bool,
    include_stage_best: bool,
) -> list[Path]:
    training_config = config.get("training", {})
    output_dir = training_config.get("output_dir")

    if output_dir is None:
        raise ValueError(
            "Cannot auto-discover checkpoints because training.output_dir is missing "
            "from the config."
        )

    output_dir = Path(output_dir)
    checkpoint_paths: list[Path] = []

    if include_overall_best:
        checkpoint_paths.append(output_dir / "best.pt")

    if include_stage_best:
        stages = training_config.get("stages", None)

        if stages is not None:
            for stage_index, stage_config in enumerate(stages, start=1):
                stage_name = stage_config.get("name", f"stage_{stage_index}")
                stage_slug = stage_checkpoint_slug(stage_index, stage_name)
                checkpoint_paths.append(output_dir / f"best_{stage_slug}.pt")
        else:
            checkpoint_paths.extend(sorted(output_dir.glob("best_stage_*.pt")))

    existing_paths = []

    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.exists():
            existing_paths.append(checkpoint_path)
        else:
            print(f"Warning: checkpoint not found, skipping: {checkpoint_path}")

    if not existing_paths:
        raise FileNotFoundError(
            f"No checkpoints found under {output_dir}. Expected files such as "
            "best.pt or best_stage_*.pt."
        )

    return existing_paths


def parse_model_specs(
    model_args: list[list[str]],
    include_overall_best: bool,
    include_stage_best: bool,
) -> list[tuple[str, str, Path, dict[str, Any]]]:
    specs = []

    for item in model_args:
        if len(item) not in [2, 3]:
            raise ValueError(
                "Each --model entry must contain either two or three values: "
                "LABEL CONFIG_PATH [CHECKPOINT_PATH]"
            )

        label = item[0]
        config_path = item[1]
        config = load_config(config_path)

        if len(item) == 3:
            checkpoint_paths = [Path(item[2])]
        else:
            checkpoint_paths = discover_checkpoints_from_config(
                config=config,
                include_overall_best=include_overall_best,
                include_stage_best=include_stage_best,
            )

        for checkpoint_path in checkpoint_paths:
            specs.append((checkpoint_label(label, checkpoint_path), config_path, checkpoint_path, config))

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
        nargs="+",
        action="append",
        required=True,
        metavar="MODEL_SPEC",
        help=(
            "Model specification. Use either two values to auto-discover checkpoints "
            "from training.output_dir, or three values to evaluate one explicit checkpoint: "
            "--model ResNet configs/resnet.yaml "
            "or --model ResNet configs/resnet.yaml checkpoints/resnet/best.pt"
        ),
    )

    parser.add_argument(
        "--no-overall-best",
        action="store_true",
        help="When checkpoint discovery is used, skip training.output_dir/best.pt.",
    )

    parser.add_argument(
        "--no-stage-best",
        action="store_true",
        help="When checkpoint discovery is used, skip best checkpoints for each training stage.",
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
    model_specs = parse_model_specs(
        model_args=args.model,
        include_overall_best=not args.no_overall_best,
        include_stage_best=not args.no_stage_best,
    )

    rows = []
    all_probs = {}

    reference_targets = None
    reference_test_loader = None
    reference_genre_to_idx = None

    loader_cache = {}

    for label, config_path, checkpoint_path, config in model_specs:
        if config_path not in loader_cache:
            loader_cache[config_path] = build_test_loader(config)

        test_loader, genre_to_idx = loader_cache[config_path]

        row, probs, targets, test_loader, genre_to_idx = evaluate_model_spec(
            label=label,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=args.threshold,
            config=config,
            test_loader=test_loader,
            genre_to_idx=genre_to_idx,
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
