"""Evaluate saved .h5 models on the test set and build the comparison table/plots."""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from config import CLASS_NAMES, MODEL_DIR, REPORT_DIR
from data import load_split


def plot_history(name):
    """Accuracy/loss curves from the history saved by train.py."""
    f = REPORT_DIR / f"{name}_history.json"
    if not f.exists():
        return
    h = json.loads(f.read_text())["history"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for a, k in zip(ax, ["accuracy", "loss"]):
        a.plot(h[k], label="train"); a.plot(h[f"val_{k}"], label="val")
        a.set_title(f"{name} - {k}"); a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(REPORT_DIR / f"{name}_history.png", dpi=120); plt.close()


def evaluate(name, test_ds, y_true):
    model = keras.models.load_model(MODEL_DIR / f"{name}.h5", compile=False)
    t0 = time.time()
    probs = model.predict(test_ds, verbose=0)
    ms_per_img = (time.time() - t0) / len(y_true) * 1000
    y_pred = probs.argmax(1)
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True,
                                zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title(f"{name} - confusion matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(REPORT_DIR / f"{name}_cm.png", dpi=120); plt.close()
    plot_history(name)
    return {"model": name, "accuracy": rep["accuracy"],
            "precision_macro": rep["macro avg"]["precision"],
            "recall_macro": rep["macro avg"]["recall"],
            "f1_macro": rep["macro avg"]["f1-score"],
            "params_M": model.count_params() / 1e6,
            "size_MB": (MODEL_DIR / f"{name}.h5").stat().st_size / 1e6,
            "ms_per_image": ms_per_img}


if __name__ == "__main__":
    REPORT_DIR.mkdir(exist_ok=True)
    test_ds = load_split("test")                       # shuffle=False keeps label order
    y_true = np.concatenate([y.numpy().argmax(1) for _, y in test_ds])
    rows = [evaluate(p.stem, test_ds, y_true) for p in sorted(MODEL_DIR.glob("*.h5"))]
    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print(df.round(4).to_string(index=False))
