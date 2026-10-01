"""Model factories: custom CNN + transfer-learning backbones."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from config import IMG_SIZE
from data import augmentation_block

N_CLASSES = 4
INPUT_SHAPE = IMG_SIZE + (3,)


def build_custom_cnn() -> keras.Model:
    """4 conv blocks (Conv-BN-ReLU x2, MaxPool, Dropout) + GAP + dense head."""
    inp = keras.Input(INPUT_SHAPE)
    x = augmentation_block()(inp)
    x = layers.Rescaling(1 / 255.0)(x)                       # 0-1 normalisation
    for filters, drop in [(32, 0.1), (64, 0.2), (128, 0.25), (256, 0.3)]:
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(drop)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(N_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name="custom_cnn")


BACKBONES = {
    "resnet50": keras.applications.ResNet50,
    "mobilenetv2": keras.applications.MobileNetV2,
    "inceptionv3": keras.applications.InceptionV3,
    "efficientnetb0": keras.applications.EfficientNetB0,
}


def preprocess(name: str, x):
    """Backbone-specific input scaling built from plain, .h5-serialisable layers.

    (keras.applications.*.preprocess_input creates raw-op layers that cannot be
    reloaded from .h5, so we rebuild the same maths with standard layers.)
    Input is raw RGB in 0-255.
    """
    if name in ("mobilenetv2", "inceptionv3"):           # -> [-1, 1]
        return layers.Rescaling(1 / 127.5, offset=-1.0)(x)
    if name == "resnet50":                               # "caffe" style: RGB->BGR, minus mean
        flip = np.zeros((1, 1, 3, 3), dtype="float32")
        for i in range(3):
            flip[0, 0, i, 2 - i] = 1.0                   # fixed 1x1 conv that reverses channels
        x = layers.Conv2D(3, 1, use_bias=False, trainable=False,
                          kernel_initializer=keras.initializers.Constant(flip))(x)
        return layers.Normalization(mean=[103.939, 116.779, 123.68], variance=[1.0] * 3)(x)
    return x                                             # EfficientNet rescales internally (0-255)


def build_transfer_model(name: str, weights="imagenet"):
    """Frozen pretrained backbone + new GAP/Dense head. Returns (model, backbone)."""
    ctor = BACKBONES[name]
    backbone = ctor(include_top=False, weights=weights, input_shape=INPUT_SHAPE)
    backbone.trainable = False
    inp = keras.Input(INPUT_SHAPE)
    x = augmentation_block()(inp)
    x = preprocess(name, x)              # backbone-specific scaling lives inside the model
    x = backbone(x, training=False)      # keep BN layers in inference mode while frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(N_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name=name), backbone


def unfreeze_top(backbone: keras.Model, n_layers: int = 30) -> None:
    """Unfreeze the last n layers (BatchNorm stays frozen) for fine-tuning."""
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
