"""O1 signal preprocessing (identical to notebook cell B1, "Fix 1").

raw guided-wave recording (n, 8, 2000) at 1 MHz
  -> zero-phase Butterworth band-pass 50-240 kHz (filtfilt)
  -> decimation by 2 (1000 samples, new Nyquist 250 kHz, so no aliasing)
  -> per-channel peak normalisation
  -> (n, 1000, 8) float32

The leakage audit found that the earlier pipeline let a month-dependent DC offset through, which a network can
use to recognise the month instead of the damage. This version refuses to return data whose mean |DC| exceeds
0.02, and rejects recordings containing NaN/inf values or a dead (all-zero) channel.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from . import config as C

_B, _A = butter(C.FILTER_ORDER, [C.UTAH_BAND[0] / (C.FS_UTAH / 2), C.UTAH_BAND[1] / (C.FS_UTAH / 2)], btype="band")


class PreprocessingError(ValueError):
    """Raised when an input recording cannot be processed safely."""


def valid_mask(gw: np.ndarray) -> np.ndarray:
    """True for recordings that are finite and have no all-zero channel. gw: (n, 8, T)."""
    gw = np.asarray(gw)
    return np.isfinite(gw).all(axis=(1, 2)) & (np.abs(gw).max(axis=2) > 0).all(axis=1)


def preprocess_utah(gw: np.ndarray) -> np.ndarray:
    """(n, 8, 2000) raw -> (n, 1000, 8) float32. Raises PreprocessingError instead of passing bad data on."""
    gw = np.asarray(gw, dtype=np.float64)
    if gw.ndim == 2:
        gw = gw[None]
    if gw.ndim != 3 or gw.shape[1] != C.N_CH or gw.shape[2] != C.RAW_SAMPLES:
        raise PreprocessingError(f"expected (n, {C.N_CH}, {C.RAW_SAMPLES}), got {gw.shape}")
    bad = ~valid_mask(gw)
    if bad.any():
        raise PreprocessingError(f"{int(bad.sum())} recording(s) contain NaN/inf or a dead channel")
    sig = filtfilt(_B, _A, gw, axis=-1)[..., :: C.STRIDE]
    peak = np.abs(sig).max(axis=-1, keepdims=True)
    out = (sig / peak).astype(np.float32)
    dc = float(np.abs(out.mean(axis=-1)).mean())
    if dc >= C.MAX_MEAN_ABS_DC:
        raise PreprocessingError(f"mean |DC| = {dc:.4f} >= {C.MAX_MEAN_ABS_DC}: filtering did not remove the offset")
    return np.transpose(out, (0, 2, 1))
