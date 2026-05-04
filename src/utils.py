"""General-purpose helper utilities."""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(mel: np.ndarray, title: str = "Mel Spectrogram",
                     save_path: str | None = None) -> None:
    """Display (and optionally save) a mel spectrogram as a colour-map."""
    plt.figure(figsize=(12, 3))
    plt.imshow(mel, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time frames")
    plt.ylabel("Mel bands")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
