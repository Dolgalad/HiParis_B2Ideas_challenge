from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from filmgenres.dataloader import create_dataloaders
from filmgenres.models import build_model
from filmgenres.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 2026))

    data_config = config["data"]

    train_loader, val_loader, test_loader, genre_to_idx = create_dataloaders(
        data_dir=data_config["data_dir"],
        image_size=data_config["image_size"],
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )

    model = build_model(
        config=config,
        num_classes=len(genre_to_idx),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        genre_to_idx=genre_to_idx,
    )

    print(f"Number of genres: {len(genre_to_idx)}")
    print(f"Genres: {genre_to_idx}")
    print(f"Training model: {config['model']['name']}")

    trainer.fit()


if __name__ == "__main__":
    main()
