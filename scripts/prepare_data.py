import os
import pandas as pd
import re
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

from pathlib import Path

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import train_test_split

# Important for reproducing the same splits
SEED=2026

TRAIN_SPLIT=0.80
VAL_SPLIT=0.10
TEST_SPLIT=0.10

DATA_FILE=Path("data/movies.csv")
ANALYSIS_TABLE_DIR=Path("report/tables")
ANALYSIS_IMG_DIR=Path("report/figures")

ANALYSIS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_IMG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR=Path("data")

POSTER_CACHE_DIR = DATA_DIR / "poster_cache"
POSTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

POSTER_TIMEOUT_SECONDS = 30
POSTER_NUM_WORKERS = max(1, os.cpu_count() or 1)
POSTER_OVERWRITE = False

"""Poster download routines
"""
def poster_cache_path(url: str, cache_dir: Path = POSTER_CACHE_DIR) -> Path:
    """
    Create a deterministic cache path for a poster URL.

    We keep the original extension when possible, but rely on a hash to avoid
    filename collisions and unsafe characters.
    """
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()

    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    return cache_dir / f"{url_hash}{suffix}"


def download_one_poster(args: tuple[str, str, bool]) -> dict:
    """
    Download one poster image.

    Returns a dictionary so multiprocessing results are easy to collect into a
    DataFrame later.
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
    cache_dir: Path = POSTER_CACHE_DIR,
    overwrite: bool = False,
    num_workers: int = POSTER_NUM_WORKERS,
) -> pd.DataFrame:
    """
    Download all unique poster URLs to cache using multiprocessing.

    This stage only downloads. Validation happens afterwards, once all downloads
    have completed.
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

def parse_genres(value: str) -> list[str]:
    """Parser for genres, genres are comma-separated. This parser returns a list of lowercase and stripped genres
    """
    if pd.isna(value):
        return []
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()

    genres = [genre.strip().lower() for genre in value.split(",")]
    genres = [genre for genre in genres if genre]

    return genres

def is_valid_genre_field(value: str) -> bool:
    genres = parse_genres(value)
    return len(genres) > 0 and all(re.match(r"^[A-Za-z][A-Za-z \-&/]*$", g) for g in genres)

def plot_genre_frequencies(genre_stats_df: pd.DataFrame, output_dir):
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
    """Grouped horizontal bar plot of genre proportions by split."""
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
    """Grouped bar plot of the number of genres per movie by split."""
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

if __name__=="__main__":
    print("Prepare and Analyse the data")

    df = pd.read_csv(DATA_FILE, header=0, engine="python")

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
            cache_dir=POSTER_CACHE_DIR,
            overwrite=POSTER_OVERWRITE,
            num_workers=POSTER_NUM_WORKERS,
    )

    # check that the cached poster images are valid after download is complete
    print("Validating cached poster images")
    poster_report = validate_cached_posters(download_report)

    poster_report.to_csv(
        ANALYSIS_TABLE_DIR / "poster_download_report.csv",
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
    genre_stats.to_csv(ANALYSIS_TABLE_DIR / "genre_frequencies.csv", index=False)
    plot_genre_frequencies(genre_stats, ANALYSIS_IMG_DIR / "genre_frequencies.png")

    # Print/plot number of genres per film
    num_genres_stats = (df["num_genres"].value_counts().sort_index().rename_axis("num_genres").reset_index(name="count"))
    num_genres_stats["proportion"] = num_genres_stats["count"] / len(df)

    print(num_genres_stats)
    num_genres_stats.to_csv(ANALYSIS_TABLE_DIR / "num_genres_distribution.csv", index=False)
    plot_num_genres_distribution(num_genres_stats, ANALYSIS_IMG_DIR / "num_genres_distribution.png")

    # create the train/val/test splits and save the corresponding rows to data/movies_{train,val,test}.csv

    train_df, temp_df = train_test_split(df, train_size=TRAIN_SPLIT, random_state=SEED, shuffle=True)
    relative_val_size = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_df, test_df = train_test_split(temp_df, train_size=relative_val_size, random_state=SEED, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train split: {len(train_df)}")
    print(f"Validation split: {len(val_df)}")
    print(f"Test split: {len(test_df)}")

    # save splits
    train_df.to_csv(DATA_DIR / "movies_train.csv", index=False)
    val_df.to_csv(DATA_DIR / "movies_val.csv", index=False)
    test_df.to_csv(DATA_DIR / "movies_test.csv", index=False)

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
        ANALYSIS_TABLE_DIR / "genre_frequencies_by_split.csv",
        index=False,
    )

    plot_genre_frequencies_by_split(
        split_genre_stats,
        ANALYSIS_IMG_DIR / "genre_frequencies_by_split.png",
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
        ANALYSIS_TABLE_DIR / "num_genres_distribution_by_split.csv",
        index=False,
    )
    
    plot_num_genres_distribution_by_split(
        split_num_genres_stats,
        ANALYSIS_IMG_DIR / "num_genres_distribution_by_split.png",
    )
