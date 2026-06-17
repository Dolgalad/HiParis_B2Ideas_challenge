from pathlib import Path
from typing import Optional
import hashlib

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def parse_genres(value: str) -> list[str]:
    """
    Parse comma-separated genre labels.
    Should match the logic used in prepare_data.py.
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()

    genres = [genre.strip().lower() for genre in value.split(",")]
    genres = [genre for genre in genres if genre]

    return genres


class PadToSquare:
    """
    Pad image to square without distorting aspect ratio.

    This is usually better for movie posters than center cropping because posters
    are tall and important content may appear near the top or bottom.
    """

    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_side = max(width, height)

        padding_left = (max_side - width) // 2
        padding_top = (max_side - height) // 2
        padding_right = max_side - width - padding_left
        padding_bottom = max_side - height - padding_top

        return ImageOps.expand(
            image,
            border=(padding_left, padding_top, padding_right, padding_bottom),
            fill=self.fill,
        )


def get_image_transform(
    split: str,
    image_size: int = 224,
    include_base: bool = True,
) -> transforms.Compose:
    base_transforms = []

    if include_base:
        base_transforms = [
            PadToSquare(fill=0),
            transforms.Resize((image_size, image_size)),
        ]

    if split == "train":
        augmentation = [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.02,
                    )
                ],
                p=0.3,
            ),
        ]
    else:
        augmentation = []

    tensor_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    return transforms.Compose(
        base_transforms + augmentation + tensor_transforms
    )

class MoviePosterDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        genre_to_idx: Optional[dict[str, int]] = None,
        transform: Optional[transforms.Compose] = None,
        poster_path_column: str = "poster_path",
        title_column: str = "Title",
        genre_column: str = "Genre",
        cache_dir: str | Path | None = None,
        cache_base_images: bool = False,
        image_size: int = 224,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        self.poster_path_column = poster_path_column
        self.title_column = title_column
        self.genre_column = genre_column
        self.transform = transform

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_base_images = cache_base_images
        self.image_size = image_size

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.base_transform = transforms.Compose([
            PadToSquare(fill=0),
            transforms.Resize((image_size, image_size)),
        ])

        if poster_path_column not in self.df.columns:
            raise ValueError(
                f"Missing column '{poster_path_column}' in {self.csv_path}. "
                "The split CSVs should include local cached poster paths."
            )

        if genre_column not in self.df.columns:
            raise ValueError(
                f"Missing column '{genre_column}' in {self.csv_path}."
            )

        self.df[genre_column] = self.df[genre_column].fillna("")

        if genre_to_idx is None:
            genres = sorted(
                {
                    genre
                    for value in self.df[genre_column]
                    for genre in parse_genres(value)
                }
            )
            genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}

        self.genre_to_idx = genre_to_idx
        self.idx_to_genre = {
            idx: genre for genre, idx in self.genre_to_idx.items()
        }
        self.num_classes = len(self.genre_to_idx)

    def __len__(self) -> int:
        return len(self.df)

    def encode_genres(self, genre_value: str) -> torch.Tensor:
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        for genre in parse_genres(genre_value):
            if genre in self.genre_to_idx:
                target[self.genre_to_idx[genre]] = 1.0

        return target

    def _cache_path_for_image(self, image_path: Path) -> Path:
        key = hashlib.md5(str(image_path.resolve()).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}_{self.image_size}.jpg"

    def _load_base_image(self, image_path: Path) -> Image.Image:
        if not self.cache_base_images or self.cache_dir is None:
            image = Image.open(image_path).convert("RGB")
            return self.base_transform(image)

        cached_path = self._cache_path_for_image(image_path)

        if cached_path.exists():
            return Image.open(cached_path).convert("RGB")

        image = Image.open(image_path).convert("RGB")
        image = self.base_transform(image)
        image.save(cached_path, quality=95)

        return image

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_path = Path(row[self.poster_path_column])

        image = self._load_base_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        target = self.encode_genres(row[self.genre_column])

        return {
            "image": image,
            "target": target,
            "poster_path": str(image_path),
            "title": row[self.title_column] if self.title_column in row else "",
        }
