from pathlib import Path

import pandas as pd

from torch.utils.data import DataLoader

from .dataset import parse_genres, MoviePosterDataset, get_image_transform

def build_genre_mapping(train_csv_path: str | Path) -> dict[str, int]:
    """
    Build genre vocabulary from the training split only.

    This avoids accidentally depending on val/test labels when defining the
    model output dimension.
    """
    train_df = pd.read_csv(train_csv_path)

    genres = sorted(
        {
            genre
            for value in train_df["Genre"].fillna("")
            for genre in parse_genres(value)
        }
    )

    return {genre: idx for idx, genre in enumerate(genres)}


def create_dataloaders(
    data_dir: str | Path = "data",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    use_cache: bool = True,
    image_normalization_mode: str = "resnet",
):
    """
    Create the train, validation and testing dataloaders. The `data_dir` argument is expected to point to a directory containing the split files 
    `movies_{train,val,test}.csv`. 
    """
    data_dir = Path(data_dir)

    train_csv = data_dir / "movies_train.csv"
    val_csv = data_dir / "movies_val.csv"
    test_csv = data_dir / "movies_test.csv"

    cache_dir = data_dir / "cache" / f"posters_{image_size}"

    genre_to_idx = build_genre_mapping(train_csv)

    train_dataset = MoviePosterDataset(
        csv_path=train_csv,
        genre_to_idx=genre_to_idx,
        transform=get_image_transform(
            "train",
            image_size=image_size,
            include_base=not use_cache,
            image_normalization_mode=image_normalization_mode,
        ),
        cache_dir=cache_dir / "train",
        cache_base_images=use_cache,
        image_size=image_size,
    )

    val_dataset = MoviePosterDataset(
        csv_path=val_csv,
        genre_to_idx=genre_to_idx,
        transform=get_image_transform(
            "val",
            image_size=image_size,
            include_base=not use_cache,
            image_normalization_mode=image_normalization_mode,
        ),
        cache_dir=cache_dir / "val",
        cache_base_images=use_cache,
        image_size=image_size,
    )

    test_dataset = MoviePosterDataset(
        csv_path=test_csv,
        genre_to_idx=genre_to_idx,
        transform=get_image_transform(
            "test",
            image_size=image_size,
            include_base=not use_cache,
            image_normalization_mode=image_normalization_mode,
        ),
        cache_dir=cache_dir / "test",
        cache_base_images=use_cache,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader, genre_to_idx
