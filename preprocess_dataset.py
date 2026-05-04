"""
Preprocess the VoxCeleb dataset: convert WAV files to mel spectrograms
and build the training pair CSV.

Usage
-----
    python preprocess_dataset.py [--src DATA/vox_indian]
                                 [--dst DATA/spectrograms]
                                 [--csv DATA/spec_dataset.csv]
"""

import argparse

from config import PATHS
from src.data import build_pair_csv, generate_spectrograms


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess VoxCeleb audio data")
    p.add_argument("--src", default=PATHS["raw_audio_dir"],
                   help="Root directory of raw WAV files")
    p.add_argument("--dst", default=PATHS["spectrogram_dir"],
                   help="Output directory for .npy spectrograms")
    p.add_argument("--csv", default=PATHS["dataset_csv"],
                   help="Output path for the pair CSV")
    return p.parse_args()


def main():
    args = parse_args()

    print("Step 1/2 — Generating mel spectrograms ...")
    generate_spectrograms(args.src, args.dst)

    print("\nStep 2/2 — Building training pair CSV ...")
    build_pair_csv(args.dst, args.csv)

    print("\nDone. You can now train the model with:\n  python train.py")


if __name__ == "__main__":
    main()
