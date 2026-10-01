"""Data loading + augmentation pipelines (tf.data)."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import BATCH_SIZE, CLASS_NAMES, DATA_DIR, IMG_SIZE, SEED


def load_split(split: str, shuffle: bool = False):
    """Load one split ('train' | 'valid' | 'test') as batched (image, one-hot label).

    Pixels are left in 0-255 here; each model rescales inside the network so the
    saved .h5 file is self-contained (the Streamlit app feeds raw pixels).
    """
    path = DATA_DIR / split
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Unzip the dataset into data/Tumour/.")
    ds = keras.utils.image_dataset_from_directory(
        path, labels="inferred", label_mode="categorical",
        class_names=CLASS_NAMES, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, shuffle=shuffle, seed=SEED,
    )
    if shuffle:  # training split only: brightness jitter (kept out of the model, see note below)
        ds = ds.map(lambda x, y: (tf.clip_by_value(tf.image.random_brightness(x, 60.0), 0.0, 255.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def augmentation_block() -> keras.Sequential:
    """Rotation, flips, zoom, shifts and contrast changes (active only in training).

    Brightness jitter is applied in load_split() instead, because Keras'
    RandomBrightness layer cannot be saved to .h5. It matters here: mean intensity
    differs a lot between classes in this dataset (glioma ~28 vs no_tumor ~60 of 255),
    so the model must not use overall brightness as a shortcut.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
    ], name="augmentation")


def class_weights(split: str = "train") -> dict:
    """Balanced class weights from folder counts (mild imbalance: glioma > others)."""
    counts = [len(list((DATA_DIR / split / c).glob("*"))) for c in CLASS_NAMES]
    total = sum(counts)
    return {i: total / (len(counts) * n) for i, n in enumerate(counts)}
