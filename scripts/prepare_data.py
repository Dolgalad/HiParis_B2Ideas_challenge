from __future__ import annotations

import os
import pandas as pd
import re
import hashlib
import argparse

from concurrent.futures import ProcessPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

from pathlib import Path

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import train_test_split

from filmgenres.dataset import parse_genres

# Important for reproducing the same splits, default values
SEED=2026

TRAIN_SPLIT=0.80
VAL_SPLIT=0.10
TEST_SPLIT=0.10

DATA_FILE="data/movies.csv"
ANALYSIS_TABLE_DIR="report/tables"
ANALYSIS_IMG_DIR="report/figures"

OUTPUT_DIR="data"

POSTER_TIMEOUT_SECONDS = 30
POSTER_NUM_WORKERS = max(1, os.cpu_count() or 1)
POSTER_OVERWRITE = False

"""Poster download routines
"""
def poster_cache_path(url: str, cache_dir: Path) -> Path:
    """
    Create a deterministic cache path for a poster URL. Original file extensions are kept but to avoid file collision a hash is used.
    """
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()

    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    return cache_dir / f"{url_hash}{suffix}"


def download_one_poster(args: tuple[str, str, bool]) -> dict:
    """
    Download one poster image.
    """
    url, output_path_str, overwrite = args
    output_path = Path(output_path_str)

    if output_path.exists() and not overwrite:
        return {
            "Poster_Url": url,
            "poster_path": str(output_path),
            "download_status": "cached",
            "download_error": "",
        }

    try:
        response = requests.get(
            url,
            timeout=POSTER_TIMEOUT_SECONDS,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "image" not in content_type:
            return {
                "Poster_Url": url,
                "poster_path": str(output_path),
                "download_status": "failed",
                "download_error": f"non-image content type: {content_type}",
            }

        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.write_bytes(response.content)
        tmp_path.replace(output_path)

        return {
            "Poster_Url": url,
            "poster_path": str(output_path),
            "download_status": "downloaded",
            "download_error": "",
        }

    except Exception as exc:
        return {
            "Poster_Url": url,
            "poster_path": str(output_path),
            "download_status": "failed",
            "download_error": repr(exc),
        }


def download_posters_to_cache(
    df: pd.DataFrame,
    cache_dir: Path,
    overwrite: bool = False,
    num_workers: int = POSTER_NUM_WORKERS,
) -> pd.DataFrame:
    """
    Download all unique poster URLs to cache using multiprocessing.
    """
    unique_urls = (
        df["Poster_Url"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    tasks = [
        (
            url,
            str(poster_cache_path(url, cache_dir)),
            overwrite,
        )
        for url in unique_urls
    ]

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_one_poster, task) for task in tasks]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading posters",
        ):
            results.append(future.result())

    return pd.DataFrame(results)


def validate_cached_poster(path: str) -> dict:
    """
    Validate a cached poster and return image metadata.
    """
    try:
        poster_path = Path(path)

        if not poster_path.exists():
            return {
                "poster_available": False,
                "poster_width": None,
                "poster_height": None,
                "poster_aspect_ratio": None,
                "poster_format": None,
                "poster_validation_error": "missing cached file",
            }

        with Image.open(poster_path) as img:
            width, height = img.size
            image_format = img.format

        return {
            "poster_available": True,
            "poster_width": width,
            "poster_height": height,
            "poster_aspect_ratio": width / height,
            "poster_format": image_format,
            "poster_validation_error": "",
        }

    except Exception as exc:
        return {
            "poster_available": False,
            "poster_width": None,
            "poster_height": None,
            "poster_aspect_ratio": None,
            "poster_format": None,
            "poster_validation_error": repr(exc),
        }


def validate_cached_posters(download_report: pd.DataFrame) -> pd.DataFrame:
    """
    Validate cached poster files after all downloads have completed.
    """
    validation_rows = []

    for row in tqdm(
        download_report.itertuples(index=False),
        total=len(download_report),
        desc="Validating cached posters",
    ):
        validation = validate_cached_poster(row.poster_path)
        validation["Poster_Url"] = row.Poster_Url
        validation_rows.append(validation)

    validation_report = pd.DataFrame(validation_rows)

    return download_report.merge(
        validation_report,
        on="Poster_Url",
        how="left",
    )

def is_valid_genre_field(value: str) -> bool:
    """
    Validation of the genres field
    """
    genres = parse_genres(value)
    return len(genres) > 0 and all(re.match(r"^[A-Za-z][A-Za-z \-&/]*$", g) for g in genres)

def plot_genre_frequencies(genre_stats_df: pd.DataFrame, output_dir):
    """
    Plot the genre frequency bar plot and save to `output_dir`.
    """
    plot_df = genre_stats_df.sort_values("movies_proportion", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["genre"], 100 * plot_df["movies_proportion"])
    plt.xlabel("Movies with genre (%)")
    plt.ylabel("Genre")
    plt.title("Genre frequency in the cleaned dataset")
    plt.tight_layout()
    plt.savefig(output_dir, dpi=300)
    plt.close()

def plot_num_genres_distribution(num_genres_stats_df: pd.DataFrame, output_dir):
    """
    Plot bar plot of the distribution of number of genres per film as save to `output_dir`.
    """
    plot_df = num_genres_stats_df.sort_values("proportion", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["num_genres"], 100 * plot_df["proportion"])
    plt.xlabel("Movies with N genres (%)")
    plt.ylabel("Genre count")
    plt.title("Genre count distribution in the cleaned dataset")
    plt.tight_layout()
    plt.savefig(output_dir, dpi=300)
    plt.close()

def compute_genre_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute genre statistics like overall proportion of movies with that genre or proportion of labels equal to that genre.
    """
    genre_counter = Counter(
        genre for genres in df["Genre"].apply(parse_genres) for genre in genres
    )

    num_movies = len(df)
    num_total_labels = sum(genre_counter.values())
    num_unique_genres = len(genre_counter)

    return (
        num_movies,
        num_unique_genres,
        num_total_labels,
        pd.DataFrame(
            [
                {
                    "genre": genre,
                    "count": count,
                    "movies_proportion": count / num_movies,
                    "label_proportion": count / num_total_labels,
                }
                for genre, count in genre_counter.items()
            ]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

def plot_genre_frequencies_by_split(split_genre_stats_df: pd.DataFrame, output_path: Path):
    """
    Grouped horizontal bar plot of genre proportions by split.
    """
    pivot_df = (
        split_genre_stats_df
        .pivot(index="genre", columns="split", values="movies_proportion")
        .fillna(0.0)
    )

    # Sort genres by train proportion for readability.
    split_order = ["train", "val", "test"]
    split_order = [s for s in split_order if s in pivot_df.columns]

    pivot_df = pivot_df.sort_values(
        by=split_order[0],
        ascending=True,
    )

    ax = (100 * pivot_df[split_order]).plot(
        kind="barh",
        figsize=(8, 6),
        width=0.8,
    )

    ax.set_xlabel("Movies with genre (%)")
    ax.set_ylabel("Genre")
    ax.set_title("Genre frequency by split")
    ax.legend(title="Split")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def compute_num_genres_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics of genre distribution.
    """
    stats = (
        df["num_genres"]
        .value_counts()
        .sort_index()
        .rename_axis("num_genres")
        .reset_index(name="count")
    )
    stats["proportion"] = stats["count"] / len(df)
    return stats

def plot_num_genres_distribution_by_split(
    split_num_genres_stats_df: pd.DataFrame,
    output_path: Path,
):
    """
    Grouped bar plot of the number of genres per movie by split.
    """
    pivot_df = (
        split_num_genres_stats_df
        .pivot(index="num_genres", columns="split", values="proportion")
        .fillna(0.0)
        .sort_index()
    )

    split_order = ["train", "val", "test"]
    split_order = [s for s in split_order if s in pivot_df.columns]

    ax = (100 * pivot_df[split_order]).plot(
        kind="bar",
        figsize=(7, 4),
        width=0.8,
    )

    ax.set_xlabel("Number of genres per movie")
    ax.set_ylabel("Movies (%)")
    ax.set_title("Distribution of number of genres per movie by split")
    ax.legend(title="Split")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        default=DATA_FILE,
        help="Path to movies CSV file (default: {DATA_FILE}).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        dest="output_dir",
        required=False,
        default=OUTPUT_DIR,
        help="Directory in which are saved the poster images and split files (default: {OUTPUT_DIR}).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=SEED,
        help=f"Seed value (default: {SEED}).",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        dest="train_split",
        required=False,
        default=TRAIN_SPLIT,
        help=f"Train split value (default: {TRAIN_SPLIT}).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        dest="val_split",
        required=False,
        default=VAL_SPLIT,
        help=f"Validation split value (default: {VAL_SPLIT}).",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        dest="test_split",
        required=False,
        default=TEST_SPLIT,
        help=f"Test split value (default: {TEST_SPLIT}).",
    )

    parser.add_argument(
        "--output-table-dir",
        type=str,
        dest="output_table_dir",
        required=False,
        default=ANALYSIS_TABLE_DIR,
        help=f"Output directory for the dataset analysis tables (default: {ANALYSIS_TABLE_DIR})."
    )

    parser.add_argument(
        "--output-img-dir",
        type=str,
        dest="output_img_dir",
        required=False,
        default=ANALYSIS_IMG_DIR,
        help=f"Output directory for the dataset analysis figures (default: {ANALYSIS_IMG_DIR})."
    )

    parser.add_argument(
        "--poster-timeout",
        type=int,
        dest="poster_timeout",
        required=False,
        default=POSTER_TIMEOUT_SECONDS,
        help=f"Poster download timeout (default: {POSTER_TIMEOUT_SECONDS})."
    )

    parser.add_argument(
        "--poster-workers",
        type=int,
        dest="poster_workers",
        required=False,
        default=POSTER_NUM_WORKERS,
        help=f"Poster download workers (default: {POSTER_NUM_WORKERS})."
    )

    parser.add_argument(
        "--poster-overwrite",
        action="store_true",
        dest="poster_overwrite",
        required=False,
        default=POSTER_OVERWRITE,
        help=f"Overwrite posters in output directory (default: {POSTER_OVERWRITE})."
    )

    return parser.parse_args()

def validate_args(args: dict[str, Any]):
    if args.train_split + args.val_split + args.test_split != 1.0:
        raise ValueError(f"Expected train/val/test split values to sum to 1 got {sum(args.train_split, args.val_split, args.test_split)}")

    # directories and paths
    args.data = Path(args.data)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_table_dir = Path(args.output_table_dir)
    args.output_table_dir.mkdir(parents=True, exist_ok=True)
    args.output_img_dir = Path(args.output_img_dir)
    args.output_img_dir.mkdir(parents=True, exist_ok=True)
    args.poster_cache_dir = args.output_dir / "poster_cache"
    args.poster_cache_dir.mkdir(parents=True, exist_ok=True)


if __name__=="__main__":
    args = parse_args()
    validate_args(args)
    print("Prepare and Analyse the data")

    df = pd.read_csv(args.data, header=0, engine="python")

    # check validity of each entry, should have a valid date, genre as a quoted comma separated list, and a poster url
    valid_date = df["Release_Date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    has_genre = df["Genre"].apply(is_valid_genre_field)
    has_poster = (
            df["Poster_Url"]
            .astype(str)
            .str.strip()
            .str.startswith("https://")
    )

    bad_rows = df[~(valid_date & has_genre & has_poster)]
    print(f"Loaded rows: {len(df)}")
    print(f"Malformed rows: {len(bad_rows)}")

    if len(bad_rows) > 0:
        print(bad_rows[["Release_Date", "Title", "Genre", "Poster_Url"]])

    df = df[valid_date & has_genre & has_poster]
    df = df.reset_index(drop=True)

    df["num_genres"] = df["Genre"].apply(parse_genres).apply(len)

    print(f"Clean rows: {len(df)}")

    # Download poster for clean rows
    print("Downloading poster images to cache")
    download_report = download_posters_to_cache(
            df,
            cache_dir=args.poster_cache_dir,
            overwrite=args.poster_overwrite,
            num_workers=args.poster_workers,
    )

    # check that the cached poster images are valid after download is complete
    print("Validating cached poster images")
    poster_report = validate_cached_posters(download_report)

    poster_report.to_csv(
        args.output_table_dir / "poster_download_report.csv",
        index=False,
    )

    df = df.merge(
        poster_report[
            [
                "Poster_Url",
                "poster_path",
                "download_status",
                "download_error",
                "poster_available",
                "poster_width",
                "poster_height",
                "poster_aspect_ratio",
                "poster_format",
                "poster_validation_error",
            ]
        ],
        on="Poster_Url",
        how="left",
    )

    unavailable_posters = df[~df["poster_available"].fillna(False)]

    print(f"Rows with unavailable or invalid posters: {len(unavailable_posters)}")

    if len(unavailable_posters) > 0:
        print(
            unavailable_posters[
                [
                    "Release_Date",
                    "Title",
                    "Poster_Url",
                    "download_status",
                    "download_error",
                    "poster_validation_error",
                ]
            ]
        )

    df = df[df["poster_available"].fillna(False)].reset_index(drop=True)



    print(f"Rows with valid cached posters: {len(df)}")

    # Print genre frequencies
    num_movies, num_unique_genres, num_total_labels, genre_stats = compute_genre_stats(df)
    print(f"Number of movies: {num_movies}")
    print(f"Number of unique genres: {num_unique_genres}")
    print(f"Total number of genre labels: {num_total_labels}")
    print(f"Average number of genres per movies: {num_total_labels / num_movies:.2f}")

    print(genre_stats)
    genre_stats.to_csv(args.output_table_dir / "genre_frequencies.csv", index=False)
    plot_genre_frequencies(genre_stats, args.output_img_dir / "genre_frequencies.png")

    # Print/plot number of genres per film
    num_genres_stats = (df["num_genres"].value_counts().sort_index().rename_axis("num_genres").reset_index(name="count"))
    num_genres_stats["proportion"] = num_genres_stats["count"] / len(df)

    print(num_genres_stats)
    num_genres_stats.to_csv(args.output_table_dir / "num_genres_distribution.csv", index=False)
    plot_num_genres_distribution(num_genres_stats, args.output_img_dir / "num_genres_distribution.png")

    # create the train/val/test splits and save the corresponding rows to data/movies_{train,val,test}.csv

    train_df, temp_df = train_test_split(df, train_size=args.train_split, random_state=args.seed, shuffle=True)
    relative_val_size = args.val_split / (args.val_split + args.test_split)
    val_df, test_df = train_test_split(temp_df, train_size=relative_val_size, random_state=args.seed, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train split: {len(train_df)}")
    print(f"Validation split: {len(val_df)}")
    print(f"Test split: {len(test_df)}")

    # save splits
    train_df.to_csv(args.output_dir / "movies_train.csv", index=False)
    val_df.to_csv(args.output_dir / "movies_val.csv", index=False)
    test_df.to_csv(args.output_dir / "movies_test.csv", index=False)

    # check that splits have similar genre distribution to the full dataset
    split_stats = []
    
    for split_name, split_df in {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }.items():
        _,_,_,stats = compute_genre_stats(split_df)
        stats["split"] = split_name
        split_stats.append(stats)
    
    split_genre_stats = pd.concat(split_stats, ignore_index=True)
    
    split_genre_stats.to_csv(
        args.output_table_dir / "genre_frequencies_by_split.csv",
        index=False,
    )

    plot_genre_frequencies_by_split(
        split_genre_stats,
        args.output_img_dir / "genre_frequencies_by_split.png",
    )

    split_num_genres_stats = []

    for split_name, split_df in {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }.items():
        stats = compute_num_genres_distribution(split_df)
        stats["split"] = split_name
        split_num_genres_stats.append(stats)
    
    split_num_genres_stats = pd.concat(
        split_num_genres_stats,
        ignore_index=True,
    )
    
    split_num_genres_stats.to_csv(
        args.output_table_dir / "num_genres_distribution_by_split.csv",
        index=False,
    )
    
    plot_num_genres_distribution_by_split(
        split_num_genres_stats,
        args.output_img_dir / "num_genres_distribution_by_split.png",
    )
