"""Train the custom CNN or a transfer-learning model.

Examples
    python src/train.py --model custom_cnn --epochs 40
    python src/train.py --model resnet50 --epochs 15 --finetune-epochs 10
    python src/train.py --model all
"""
import argparse
import json
import time

from tensorflow import keras

from config import MODEL_DIR, REPORT_DIR, SEED
from data import class_weights, load_split
from models import BACKBONES, build_custom_cnn, build_transfer_model, unfreeze_top

ALL_MODELS = ["custom_cnn", *BACKBONES]


def make_callbacks(name: str):
    """EarlyStopping + ModelCheckpoint on val_loss, plus LR reduction."""
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_DIR / f"{name}.h5", monitor="val_loss",
                                        save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
    ]


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy",
                  metrics=["accuracy"])


def train_one(name, epochs, finetune_epochs, weights="imagenet"):
    keras.utils.set_random_seed(SEED)
    train_ds, val_ds = load_split("train", shuffle=True), load_split("valid")
    cw = class_weights()

    if name == "custom_cnn":
        model = build_custom_cnn()
        compile_model(model, 1e-3)
        t0 = time.time()
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                         class_weight=cw, callbacks=make_callbacks(name)).history
    else:
        model, backbone = build_transfer_model(name, weights)
        compile_model(model, 1e-3)                       # stage 1: train the new head only
        t0 = time.time()
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                         class_weight=cw, callbacks=make_callbacks(name)).history
        if finetune_epochs > 0:                          # stage 2: fine-tune top layers
            unfreeze_top(backbone)
            compile_model(model, 1e-5)                   # low LR protects pretrained weights
            h2 = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs,
                           class_weight=cw, callbacks=make_callbacks(name)).history
            for k, v in h2.items():
                hist[k] = list(hist[k]) + list(v)

    secs = time.time() - t0
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f"{name}_history.json").write_text(json.dumps(
        {"history": {k: [float(x) for x in v] for k, v in hist.items()}, "train_seconds": secs}))
    print(f"[{name}] finished in {secs/60:.1f} min -> {MODEL_DIR / (name + '.h5')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="custom_cnn", choices=[*ALL_MODELS, "all"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    a = p.parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    w = None if a.weights == "none" else "imagenet"
    for m in (ALL_MODELS if a.model == "all" else [a.model]):
        train_one(m, a.epochs, a.finetune_epochs, w)
