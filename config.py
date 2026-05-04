"""
Central configuration for the Voice Authentication system.
All hyperparameters, paths, and model settings are defined here.
"""

# ---------------------------------------------------------------------------
# Audio / Mel-spectrogram parameters
# ---------------------------------------------------------------------------
AUDIO_CONFIG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "n_mels": 80,
    "frame_length": 1024,
    "frame_shift": 256,       # hop length
    "mel_fmin": 0,
    "mel_fmax": 8000,
    "top_db": 60,             # silence trimming threshold
    "spec_width": 450,        # fixed temporal width after pad/crop
    "spec_height": 80,        # mel bands (== n_mels)
}

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "embedding_dim": 512,
    "dropout_rate": 0.3,
    "conv_blocks": [
        {"filters": 32, "kernel_size": (4, 20), "num_layers": 3},
        {"filters": 64, "kernel_size": (4, 20), "num_layers": 3},
        {"filters": 32, "kernel_size": (4, 20), "num_layers": 3},
    ],
    "pool_size": (2, 2),
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN_CONFIG = {
    "batch_size": 128,
    "epochs": 10,
    "learning_rate": 1e-2,
    "loss": "binary_crossentropy",
    "metrics": ["accuracy"],
}

# ---------------------------------------------------------------------------
# Paths  (override via environment variables or CLI arguments)
# ---------------------------------------------------------------------------
PATHS = {
    "raw_audio_dir": "data/vox_indian",
    "spectrogram_dir": "data/spectrograms",
    "dataset_csv": "data/spec_dataset.csv",
    "model_checkpoint": "checkpoints/siamese_model.h5",
    "embedder_checkpoint": "checkpoints/embedder.h5",
    "tail_checkpoint": "checkpoints/tail.h5",
}
