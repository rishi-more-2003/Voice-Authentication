"""
Train the Siamese speaker-verification model.

Usage
-----
    python train.py [--csv DATA_CSV] [--epochs N] [--batch-size B] [--lr LR]
                    [--checkpoint PATH]
"""

import argparse
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from config import PATHS, TRAIN_CONFIG
from src.data import pair_generator
from src.model import build_embedding_model, build_siamese_model


def parse_args():
    p = argparse.ArgumentParser(description="Train the Siamese model")
    p.add_argument("--csv", default=PATHS["dataset_csv"],
                   help="Path to the spectrogram-pair CSV")
    p.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    p.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"])
    p.add_argument("--checkpoint", default=PATHS["model_checkpoint"],
                   help="Where to save the trained model (.h5)")
    return p.parse_args()


def main():
    args = parse_args()

    # GPU memory growth
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Build model
    embedder = build_embedding_model()
    model = build_siamese_model(embedder)
    model.compile(optimizer=Adam(learning_rate=args.lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    print("\n" + "=" * 60)
    print("  Siamese Network — Model Summary")
    print("=" * 60)
    model.summary()
    print()

    embedder.summary()
    print()

    # Data generator
    import pandas as pd
    data = pd.read_csv(args.csv)
    steps_per_epoch = len(data) // args.batch_size

    gen = pair_generator(csv_path=args.csv, batch_size=args.batch_size)

    # Callbacks
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="logs"),
        tf.keras.callbacks.ModelCheckpoint(
            args.checkpoint, save_best_only=True,
            monitor="accuracy", mode="max",
        ),
    ]

    # Train
    model.fit(gen,
              epochs=args.epochs,
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)

    # Save final artefacts
    model.save(args.checkpoint)
    embedder.save(PATHS["embedder_checkpoint"])
    print(f"\nModel saved to {args.checkpoint}")
    print(f"Embedder saved to {PATHS['embedder_checkpoint']}")


if __name__ == "__main__":
    main()
