"""Dataset-level evaluation utilities for the Brain Tumor MRI EDA project."""

from pathlib import Path

import pandas as pd

from Config import CLASS_NAMES, SPLITS
from Data import count_images_by_class


def build_split_summary(data_dir: Path) -> pd.DataFrame:
    """Build a table containing image counts for each dataset split."""

    rows = []

    for split in SPLITS:
        counts = count_images_by_class(data_dir, split)

        row = {
            "split": split,
            **counts,
            "total": sum(counts.values()),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def build_class_summary(data_dir: Path) -> pd.DataFrame:
    """Build a class-count summary across train, validation and test splits."""

    rows = []

    for class_name in CLASS_NAMES:
        row = {"class": class_name}

        total = 0

        for split in SPLITS:
            counts = count_images_by_class(data_dir, split)
            count = counts.get(class_name, 0)

            row[split] = count
            total += count

        row["total"] = total
        rows.append(row)

    return pd.DataFrame(rows)


def print_dataset_summary(data_dir: Path) -> None:
    """Print dataset split and class summaries."""

    split_summary = build_split_summary(data_dir)
    class_summary = build_class_summary(data_dir)

    print("\nDataset Split Summary")
    print("=" * 60)
    print(split_summary.to_string(index=False))

    print("\nClass Summary")
    print("=" * 60)
    print(class_summary.to_string(index=False))


if __name__ == "__main__":
    from Config import ROOT

    data_dir = ROOT / "Tumour"

    print_dataset_summary(data_dir)
