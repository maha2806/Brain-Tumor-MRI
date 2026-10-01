"""Shared configuration for the Brain Tumor MRI project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Tumour"          # expects train/ valid/ test/ sub-folders
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
# Alphabetical order == order used by image_dataset_from_directory
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
