"""CNN model builders for multi-label digit classification."""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_model(input_shape=(64, 64, 1)):
    """
    Baseline CNN, identical to the architecture in cnn.ipynb.

    Sigmoid output (not softmax): each of the 10 output neurons answers
    independently whether that digit class is present in the image.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs

    x = layers.Conv2D(32, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(10, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="multi_label_cnn")


def build_improved_cnn_model(input_shape=(64, 64, 1)):
    """
    Improved CNN: adds batch normalization, an extra conv block, and
    heavier regularization compared to the baseline. Starting point for
    the "propose improvements" part of the lab exercise -- tune further
    as needed.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(10, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="multi_label_cnn_improved")


MODEL_BUILDERS = {
    "baseline": build_cnn_model,
    "improved": build_improved_cnn_model,
}
