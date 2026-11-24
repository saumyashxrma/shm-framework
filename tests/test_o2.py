"""REQ-F-07..10 O2: physics features, monotone diagnosis, forecasting, conformal intervals."""
import numpy as np

from shm import config as C
from shm.conformal import conformal_q, coverage, interval
from shm.features_o2 import phys_features
from shm.forecast import lin_gp_forecast
from shm.model_o2 import CrackDiagnoser, causal_monotone
from shm.pipeline import synthetic_specimens


def _signals(n=5, L=4000):
    t = np.arange(L)
    base = np.sin(2 * np.pi * t / 80) * np.exp(-((t - 1000) / 200.0) ** 2)
    rx = np.stack([base * (1 - 0.1 * k) + 0.05 * k * np.roll(base, 600) for k in range(n)])
    return rx, np.ones(n)


def test_features_shape_names_and_reference_row():
    rx, tp = _signals()
    F = phys_features(rx, tp)
    assert F.shape == (5, len(C.PHYS_NAMES)) and np.isfinite(F).all()
    assert np.allclose(F[0], 0.0, atol=1e-9)              # the reference recording has no change


def test_features_track_amplitude_loss():
    rx, tp = _signals()
    F = phys_features(rx, tp)
    assert np.all(np.diff(F[:, C.PHYS_NAMES.index("log_amp")]) < 0)


def test_causal_monotone():
    assert causal_monotone([0, 1, 0.5, 2, 1.5]).tolist() == [0, 1, 1, 2, 2]


def test_diagnoser_outputs_non_negative_monotone_estimates():
    D = synthetic_specimens()
    m = CrackDiagnoser(1, 50).fit(D, C.TRAIN_SPECS)
    est = m.predict(D["T7"]["cycles"], D["T7"]["phys"])
    assert (est >= 0).all() and np.all(np.diff(est) >= 0)


def test_linear_gp_forecast_extrapolates_a_linear_trend():
    c = np.linspace(0, 50000, 8)
    a = 1e-4 * c
    f = lin_gp_forecast(c, a, [60000, 70000])
    assert np.allclose(f, [6.0, 7.0], atol=0.3)


def test_conformal_quantile_and_coverage():
    assert conformal_q(np.arange(1, 10)) == 9.0           # ceil(10 * 0.9) = 9th smallest
    rng = np.random.default_rng(0)
    q = conformal_q(rng.normal(size=500))
    assert 0.86 < coverage(rng.normal(size=5000), np.zeros(5000), q) < 0.94
    lo, hi = interval([1.0, 5.0], 3.08)
    assert lo.tolist() == [0.0, 1.92] and hi.tolist() == [4.08, 8.08]
