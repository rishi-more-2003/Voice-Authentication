"""
Audio preprocessing pipeline.

Converts raw WAV files to mel spectrograms suitable for the Siamese
network. Key optimisations include globally cached Hann windows and
mel filter-banks to avoid redundant computation across samples.
"""

import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read

from config import AUDIO_CONFIG

# Global caches to avoid recomputing filter-banks / windows per sample.
_mel_basis: dict = {}
_hann_window: dict = {}


def load_wav(path: str):
    """Read a WAV file and return ``(data, sampling_rate)``."""
    sampling_rate, data = read(path)
    return data, sampling_rate


def dynamic_range_compression(x: torch.Tensor, C: float = 1.0,
                              clip_val: float = 1e-5) -> torch.Tensor:
    """Log-compress a spectrogram to limit dynamic range."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize(magnitudes: torch.Tensor) -> torch.Tensor:
    """Apply dynamic-range compression as spectral normalisation."""
    return dynamic_range_compression(magnitudes)


def mel_spectrogram(y: torch.Tensor, n_fft: int, n_mels: int, sr: int,
                    hop_size: int, win_size: int, fmin: int, fmax: int,
                    center: bool = False) -> torch.Tensor:
    """Compute a mel spectrogram from a waveform tensor.

    Uses globally cached Hann windows and mel filter-banks so that
    repeated calls do not rebuild these matrices.
    """
    global _mel_basis, _hann_window

    device_key = str(y.device)
    mel_key = f"{fmax}_{device_key}"

    if mel_key not in _mel_basis:
        mel = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels,
                             fmin=fmin, fmax=fmax)
        _mel_basis[mel_key] = torch.from_numpy(mel).float().to(y.device)

    if device_key not in _hann_window:
        _hann_window[device_key] = torch.hann_window(win_size).to(y.device)

    y = F.pad(y.unsqueeze(1),
              (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
              mode="reflect")
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                      window=_hann_window[device_key],
                      center=center, pad_mode="reflect",
                      normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(_mel_basis[mel_key], spec)
    spec = spectral_normalize(spec)
    return spec


def wav_to_mel(path: str, cfg: dict | None = None) -> tuple:
    """Load a WAV file and return ``(audio_np, mel_np)``.

    Parameters
    ----------
    path : str
        Path to a ``.wav`` file.
    cfg : dict, optional
        Audio configuration dict (defaults to ``AUDIO_CONFIG``).

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
    """
    if cfg is None:
        cfg = AUDIO_CONFIG

    try:
        audio, _ = librosa.load(path, sr=cfg["sample_rate"])
    except Exception:
        return None, None

    audio = librosa.effects.trim(audio, top_db=cfg["top_db"],
                                 frame_length=512, hop_length=256)[0]
    audio = librosa.util.normalize(audio) * 0.95

    mel = mel_spectrogram(
        y=torch.from_numpy(audio).float().unsqueeze(0),
        n_fft=cfg["n_fft"],
        n_mels=cfg["n_mels"],
        sr=cfg["sample_rate"],
        hop_size=cfg["frame_shift"],
        win_size=cfg["frame_length"],
        fmin=cfg["mel_fmin"],
        fmax=cfg["mel_fmax"],
    ).squeeze(0).T.numpy()

    return audio, mel.T


# ------------------------------------------------------------------
# Padding / cropping utilities
# ------------------------------------------------------------------

def pad_spectrogram(arr: np.ndarray, target_shape: tuple,
                    mode: str = "mean") -> np.ndarray:
    """Zero- or mean-pad a spectrogram to ``target_shape``."""
    pad_width = [(0, max(0, target_shape[0] - arr.shape[0])),
                 (0, max(0, target_shape[1] - arr.shape[1]))]
    return np.pad(arr, pad_width, mode=mode)


def pad_or_crop(spec: np.ndarray, width: int | None = None,
                mode: str = "mean") -> np.ndarray:
    """Pad or crop a spectrogram to a fixed temporal width."""
    if width is None:
        width = AUDIO_CONFIG["spec_width"]
    height = AUDIO_CONFIG["spec_height"]

    if spec.shape[1] < width:
        return pad_spectrogram(spec, (height, width), mode=mode)
    return spec[:, :width]


def get_mel(path: str) -> np.ndarray:
    """End-to-end: WAV path -> fixed-size mel spectrogram matrix."""
    _, mel = wav_to_mel(path)
    return pad_or_crop(mel, mode="mean")
