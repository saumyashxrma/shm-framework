"""REQ-F-01 preprocessing contract: shape, normalisation, DC removal, rejection of invalid input."""
import numpy as np
import pytest

from shm import config as C
from shm.preprocess import PreprocessingError, preprocess_utah, valid_mask


def _raw(n=3, seed=0, dc=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(C.RAW_SAMPLES) / C.FS_UTAH
    burst = np.sin(2 * np.pi * 100e3 * t) * np.exp(-((t - 2e-4) / 5e-5) ** 2)
    return burst[None, None, :] * rng.uniform(0.5, 2.0, (n, C.N_CH, 1)) + rng.normal(0, 0.01, (n, C.N_CH, C.RAW_SAMPLES)) + dc


def test_output_shape_and_dtype():
    out = preprocess_utah(_raw(4))
    assert out.shape == (4, C.N_SAMP, C.N_CH) and out.dtype == np.float32


def test_peak_normalised_per_channel():
    out = preprocess_utah(_raw(2))
    assert np.allclose(np.abs(out).max(axis=1), 1.0, atol=1e-6)


def test_month_dependent_dc_offset_is_removed():
    a, b = preprocess_utah(_raw(2, dc=0.0)), preprocess_utah(_raw(2, dc=5.0))
    assert np.abs(a.mean(axis=1)).mean() < C.MAX_MEAN_ABS_DC
    assert np.abs(b.mean(axis=1)).mean() < C.MAX_MEAN_ABS_DC


def test_rejects_nan_and_dead_channel():
    x = _raw(3)
    x[1, 2, 10] = np.nan
    with pytest.raises(PreprocessingError):
        preprocess_utah(x)
    y = _raw(3)
    y[0, 5, :] = 0.0
    assert valid_mask(y).tolist() == [False, True, True]
    with pytest.raises(PreprocessingError):
        preprocess_utah(y)


def test_rejects_wrong_shape():
    with pytest.raises(PreprocessingError):
        preprocess_utah(np.zeros((2, 8, 1500)))
