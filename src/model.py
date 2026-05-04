"""
Siamese network architecture for speaker verification.

The model consists of two modules:

* **Embedding (Tail)** — A CNN that maps an 80x450 mel spectrogram to a
  compact 512-dimensional embedding vector.
* **Siamese Head** — Computes the L1 (Manhattan) distance between two
  embeddings and passes it through a sigmoid classifier.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, Flatten, Input, Layer, MaxPooling2D,
)
from tensorflow.keras.models import Model

from config import AUDIO_CONFIG, MODEL_CONFIG


# -----------------------------------------------------------------------
# L1 Distance Layer
# -----------------------------------------------------------------------

class L1Dist(Layer):
    """Custom Keras layer that computes element-wise L1 distance."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# -----------------------------------------------------------------------
# Embedding (Tail) model
# -----------------------------------------------------------------------

def build_embedding_model(input_shape: tuple | None = None) -> Model:
    """Build the tail / encoder CNN that produces speaker embeddings.

    Parameters
    ----------
    input_shape : tuple, optional
        ``(height, width, channels)`` — defaults to ``(80, 450, 1)``.

    Returns
    -------
    keras.Model
    """
    if input_shape is None:
        h = AUDIO_CONFIG["spec_height"]
        w = AUDIO_CONFIG["spec_width"]
        input_shape = (h, w, 1)

    cfg = MODEL_CONFIG
    x = inp = Input(shape=input_shape, name="input_spec")

    for block in cfg["conv_blocks"]:
        for _ in range(block["num_layers"]):
            x = Conv2D(block["filters"], block["kernel_size"],
                       activation="relu")(x)
        x = Dropout(cfg["dropout_rate"])(x)
        x = MaxPooling2D(cfg["pool_size"], padding="same")(x)

    x = Flatten()(x)
    x = Dense(cfg["embedding_dim"], activation="sigmoid")(x)

    return Model(inputs=[inp], outputs=[x], name="Embedding")


# -----------------------------------------------------------------------
# Siamese (Head) model
# -----------------------------------------------------------------------

def build_siamese_model(embedder: Model | None = None) -> Model:
    """Build the full Siamese verification network.

    Parameters
    ----------
    embedder : keras.Model, optional
        Pre-built embedding model. If ``None`` one is created internally.

    Returns
    -------
    keras.Model
    """
    if embedder is None:
        embedder = build_embedding_model()

    h = AUDIO_CONFIG["spec_height"]
    w = AUDIO_CONFIG["spec_width"]

    input_a = Input(name="input_a", shape=(h, w))
    input_b = Input(name="input_b", shape=(h, w))

    distance = L1Dist(name="l1_distance")(
        embedder(input_a), embedder(input_b)
    )
    output = Dense(1, activation="sigmoid")(distance)

    return Model(inputs=[input_a, input_b], outputs=output,
                 name="SiameseNetwork")


# -----------------------------------------------------------------------
# Convenience loaders
# -----------------------------------------------------------------------

def load_siamese_model(path: str) -> Model:
    """Load a saved Siamese model from an ``.h5`` file."""
    return keras.models.load_model(
        path, custom_objects={"L1Dist": L1Dist}
    )


def load_embedder(path: str) -> Model:
    """Load a saved embedding (tail) model."""
    return keras.models.load_model(path)
