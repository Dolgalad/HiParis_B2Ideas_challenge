#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def read_history_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = {"epoch", "val_map"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}"
        )

    df["epoch"] = pd.to_numeric(df["epoch"])
    df["val_map"] = pd.to_numeric(df["val_map"])

    return df.sort_values("epoch")


def plot_val_map(
    csv_paths: list[str],
    labels: list[str],
    title: str,
    output_path: str | None = None,
) -> None:
    if len(csv_paths) != len(labels):
        raise ValueError(
            f"Expected one label per CSV file, got "
            f"{len(csv_paths)} CSV files and {len(labels)} labels."
        )

    plt.figure(figsize=(9, 5.5))

    for csv_path, label in zip(csv_paths, labels):
        df = read_history_csv(csv_path)

        plt.plot(
            df["epoch"],
            df["val_map"],
            linestyle="--",
            marker=None,
            label=f"{label}",
        )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MAP")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot validation MAP from one or more training history CSV files."
        )
    )

    parser.add_argument(
        "csv_paths",
        nargs="+",
        help="Path(s) to history CSV files.",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Legend label for each CSV file, in the same order.",
    )

    parser.add_argument(
        "--title",
        required=True,
        help="Title of the plot.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the plot, e.g. figures/val_map_comparison.png.",
    )

    args = parser.parse_args()

    plot_val_map(
        csv_paths=args.csv_paths,
        labels=args.labels,
        title=args.title,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
