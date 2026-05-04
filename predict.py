"""
Run speaker-verification inference on two audio files.

Usage
-----
    python predict.py --audio1 path/to/clip_a.wav --audio2 path/to/clip_b.wav
                      [--model checkpoints/siamese_model.h5]
                      [--threshold 0.5]
"""

import argparse

import numpy as np
import tensorflow as tf

from config import AUDIO_CONFIG, PATHS
from src.model import load_siamese_model
from src.preprocessing import get_mel


def parse_args():
    p = argparse.ArgumentParser(
        description="Verify whether two audio clips belong to the same speaker"
    )
    p.add_argument("--audio1", required=True, help="Path to first .wav file")
    p.add_argument("--audio2", required=True, help="Path to second .wav file")
    p.add_argument("--model", default=PATHS["model_checkpoint"],
                   help="Path to trained Siamese model (.h5)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold (default: 0.5)")
    return p.parse_args()


def main():
    args = parse_args()

    # GPU memory growth
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"Loading model from {args.model} ...")
    model = load_siamese_model(args.model)

    print(f"Processing {args.audio1} ...")
    mel1 = get_mel(args.audio1)

    print(f"Processing {args.audio2} ...")
    mel2 = get_mel(args.audio2)

    h, w = AUDIO_CONFIG["spec_height"], AUDIO_CONFIG["spec_width"]
    mel1 = mel1.reshape(-1, h, w)
    mel2 = mel2.reshape(-1, h, w)

    score = model.predict([mel1, mel2])[0][0]
    match = score >= args.threshold

    print()
    print("=" * 50)
    print(f"  Similarity score : {score:.4f}")
    print(f"  Threshold        : {args.threshold}")
    print(f"  Result           : {'SAME speaker' if match else 'DIFFERENT speakers'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
