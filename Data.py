"""Data loading and basic image-statistics utilities for the Brain Tumor MRI EDA project."""

from pathlib import Path
from PIL import Image
import numpy as np

from Config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    THUMBNAIL_SIZE,
)


def get_split_path(data_dir: Path, split: str) -> Path:
    """Return the path for a dataset split.

    Expected structure:
        Tumour/
        ├── train/
        ├── valid/
        └── test/
    """
    path = data_dir / split

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}"
        )

    return path


def get_image_files(data_dir: Path, split: str) -> list[Path]:
    """Return all image files from one dataset split."""
    split_path = get_split_path(data_dir, split)

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    return sorted(
        file
        for file in split_path.rglob("*")
        if file.is_file() and file.suffix.lower() in image_extensions
    )


def load_image(path: Path) -> Image.Image:
    """Load an MRI image as an RGB PIL image."""
    with Image.open(path) as image:
        return image.convert("RGB")


def get_image_dimensions(path: Path) -> tuple[int, int]:
    """Return image dimensions as (width, height)."""
    with Image.open(path) as image:
        return image.size


def calculate_image_statistics(
    path: Path,
    thumbnail_size: int = THUMBNAIL_SIZE,
) -> dict:
    """Calculate the image-level statistics used for EDA.

    Statistics include:
    - width and height
    - image mode
    - mean intensity
    - intensity standard deviation
    - 95th-percentile intensity
    - foreground fraction
    """
    image = load_image(path)

    width, height = image.size

    thumbnail = image.resize(
        (thumbnail_size, thumbnail_size)
    )

    array = np.asarray(thumbnail, dtype=np.float32)

    # Convert RGB image to grayscale intensity.
    intensity = (
        0.299 * array[:, :, 0]
        + 0.587 * array[:, :, 1]
        + 0.114 * array[:, :, 2]
    )

    mean_intensity = float(np.mean(intensity))
    std_intensity = float(np.std(intensity))
    p95_intensity = float(np.percentile(intensity, 95))

    # Foreground pixels are pixels above zero intensity.
    foreground_fraction = float(
        np.mean(intensity > 0)
    )

    return {
        "width": width,
        "height": height,
        "mode": image.mode,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "p95_intensity": p95_intensity,
        "foreground_fraction": foreground_fraction,
    }


def validate_image_dimensions(path: Path) -> bool:
    """Check whether an image matches the expected dataset dimensions."""
    return get_image_dimensions(path) == IMAGE_SIZE


def count_images_by_class(
    data_dir: Path,
    split: str,
) -> dict[str, int]:
    """Count images for each tumour class in a dataset split."""
    split_path = get_split_path(data_dir, split)

    counts = {}

    for class_name in CLASS_NAMES:
        class_path = split_path / class_name

        if class_path.exists():
            counts[class_name] = sum(
                1
                for file in class_path.rglob("*")
                if file.is_file()
                and file.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif",
                    ".tiff",
                }
            )
        else:
            counts[class_name] = 0

    return counts
