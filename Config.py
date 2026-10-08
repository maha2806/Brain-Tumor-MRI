"""Configuration for the Brain Tumor MRI EDA project."""

from pathlib import Path


# Project root
ROOT = Path(__file__).resolve().parent


# Dataset information
DATASET_NAME = "Tumour"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

NUM_CLASSES = len(CLASS_NAMES)


# Dataset characteristics
IMAGE_SIZE = (640, 640)
IMAGE_MODE = "RGB"

# Dataset split folders
SPLITS = [
    "train",
    "valid",
    "test"
]


# EDA thumbnail size
THUMBNAIL_SIZE = 64


# Reproducibility
SEED = 42
