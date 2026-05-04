"""
Dataset utilities for training the Siamese speaker-verification model.

Provides helpers for:
* Generating mel-spectrogram ``.npy`` files from a directory of WAV files.
* Building a CSV of spectrogram pairs with same/different speaker labels.
* A batch generator that feeds paired spectrograms to ``model.fit()``.
"""

from __future__ import annotations

import os
from typing import Generator

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import AUDIO_CONFIG, PATHS, TRAIN_CONFIG
from src.preprocessing import pad_or_crop, wav_to_mel


# -----------------------------------------------------------------------
# Spectrogram generation
# -----------------------------------------------------------------------

def generate_spectrograms(src_dir: str, dst_dir: str) -> None:
    """Convert every WAV in *src_dir* to a mel-spectrogram ``.npy`` file.

    The directory tree ``src_dir/<speaker>/<video>/<clip>.wav`` is mirrored
    under *dst_dir* with ``.npy`` files instead of ``.wav``.
    """
    os.makedirs(dst_dir, exist_ok=True)

    for speaker_id in tqdm(os.listdir(src_dir), desc="Speakers"):
        speaker_dir = os.path.join(src_dir, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue

        mel_speaker_dir = os.path.join(dst_dir, speaker_id)
        os.makedirs(mel_speaker_dir, exist_ok=True)

        for video_id in os.listdir(speaker_dir):
            video_dir = os.path.join(speaker_dir, video_id)
            if not os.path.isdir(video_dir):
                continue

            for wav_file in os.listdir(video_dir):
                if not wav_file.endswith(".wav"):
                    continue
                wav_path = os.path.join(video_dir, wav_file)
                _, mel = wav_to_mel(wav_path)
                if mel is None:
                    continue
                mel = pad_or_crop(mel, mode="mean")
                npy_name = wav_file.replace(".wav", ".npy")
                np.save(os.path.join(mel_speaker_dir,
                                     f"{video_id}_{npy_name}"), mel)


# -----------------------------------------------------------------------
# Pair CSV creation
# -----------------------------------------------------------------------

def build_pair_csv(spec_dir: str, output_csv: str | None = None) -> pd.DataFrame:
    """Create a CSV of ``(mel_1, mel_2, same)`` pairs for training.

    For every pair of speakers the function produces both positive pairs
    (same speaker) and negative pairs (different speakers).
    """
    if output_csv is None:
        output_csv = PATHS["dataset_csv"]

    rows: list[dict] = []
    speakers = sorted(os.listdir(spec_dir))

    for s1 in tqdm(speakers, desc="Building pairs"):
        s1_dir = os.path.join(spec_dir, s1)
        if not os.path.isdir(s1_dir):
            continue
        s1_files = [os.path.join(s1_dir, f)
                    for f in os.listdir(s1_dir) if f.endswith(".npy")]

        # Positive pairs (same speaker)
        for i, f1 in enumerate(s1_files):
            for f2 in s1_files[i + 1:]:
                rows.append({"mel_1": f1, "mel_2": f2, "same": 1})

        # Negative pairs (different speaker)
        for s2 in speakers:
            if s2 == s1:
                continue
            s2_dir = os.path.join(spec_dir, s2)
            if not os.path.isdir(s2_dir):
                continue
            s2_files = [os.path.join(s2_dir, f)
                        for f in os.listdir(s2_dir) if f.endswith(".npy")]
            for f1 in s1_files[:2]:
                for f2 in s2_files[:2]:
                    rows.append({"mel_1": f1, "mel_2": f2, "same": 0})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} pairs to {output_csv}")
    return df


# -----------------------------------------------------------------------
# Training data generator
# -----------------------------------------------------------------------

def _load_mel(path: str) -> np.ndarray:
    mel = np.load(path)
    mel = pad_or_crop(mel, mode="mean")
    return mel


def pair_generator(csv_path: str | None = None,
                   batch_size: int | None = None
                   ) -> Generator[tuple, None, None]:
    """Yield ``([X1, X2], y)`` batches for ``model.fit()``.

    Uses an in-memory cache so each ``.npy`` file is loaded at most once.
    """
    if csv_path is None:
        csv_path = PATHS["dataset_csv"]
    if batch_size is None:
        batch_size = TRAIN_CONFIG["batch_size"]

    data = pd.read_csv(csv_path)
    h = AUDIO_CONFIG["spec_height"]
    w = AUDIO_CONFIG["spec_width"]
    cache: dict[str, np.ndarray] = {}

    while True:
        batch = data.sample(batch_size)
        X1 = np.zeros((batch_size, h, w))
        X2 = np.zeros((batch_size, h, w))
        y = batch["same"].values

        for i, (_, row) in enumerate(batch.iterrows()):
            if row["mel_1"] not in cache:
                cache[row["mel_1"]] = _load_mel(row["mel_1"])
            if row["mel_2"] not in cache:
                cache[row["mel_2"]] = _load_mel(row["mel_2"])
            X1[i] = cache[row["mel_1"]]
            X2[i] = cache[row["mel_2"]]

        yield [X1, X2], y
