"""O1 Stage-2 model: CNN-BiLSTM damage-level regression, 3-seed ensemble per leave-damage-out fold.

Architecture (cell K7; about 87 k parameters): wave input (1000, 8) -> GaussianNoise -> SpatialDropout1D ->
2 x [Conv1D, BatchNorm, MaxPool] -> MaxPool -> BiLSTM(64) -> Dropout, concatenated with a normalised
(temperature, humidity) branch -> Dense(64) -> linear damage level. Trained with Huber loss (delta = 1),
class-balanced weights and time-stretch/shift augmentation.

TensorFlow is imported lazily so that the rest of the package (and the CI test suite) runs without it.
Model files: models/v3_level_fold{A|B|C}_seed{42|123|2024}.keras (exported from the Kaggle run).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C


def model_path(fold: str, seed: int, models_dir: Path = C.MODELS_DIR) -> Path:
    return Path(models_dir) / f"v3_level_fold{fold}_seed{seed}.keras"


def available(fold: str = "A", models_dir: Path = C.MODELS_DIR) -> bool:
    """True when all three seed models of the fold exist and TensorFlow is importable."""
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return all(model_path(fold, s, models_dir).exists() for s in C.SEEDS)


class LevelEnsemble:
    """Mean of the three seed models of one fold."""

    def __init__(self, fold: str = "A", models_dir: Path = C.MODELS_DIR):
        import tensorflow as tf
        self.fold = fold
        self.models = [tf.keras.models.load_model(model_path(fold, s, models_dir)) for s in C.SEEDS]

    def predict(self, waves, env, batch: int = 256) -> np.ndarray:
        """waves: (n, 1000, 8) preprocessed; env: (n, 2) [temperature deg C, humidity %]. Returns level (n,)."""
        waves = np.asarray(waves, np.float32)
        env = np.asarray(env, np.float32)
        if waves.ndim != 3 or waves.shape[1:] != (C.N_SAMP, C.N_CH):
            raise ValueError(f"waves must be (n, {C.N_SAMP}, {C.N_CH}), got {waves.shape}")
        if env.shape != (len(waves), 2):
            raise ValueError("env must be (n, 2)")
        preds = []
        for m in self.models:
            out = [np.asarray(m.predict_on_batch({"wave_input": waves[s:s + batch], "env_input": env[s:s + batch]})).ravel()
                   for s in range(0, len(waves), batch)]
            preds.append(np.concatenate(out))
        return np.mean(preds, axis=0)
