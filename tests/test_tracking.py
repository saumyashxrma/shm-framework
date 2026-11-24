"""REQ-F-04/05 decision layer: thresholds, causal smoothing, cumulative tracking, change detection."""
import numpy as np

from shm.tracking import causal_median, detect, level_to_mech, level_to_sev, sev_from_tags, track


def test_severity_thresholds_at_the_edges():
    assert level_to_sev([0, 4.49, 4.5, 9.49, 9.5, 13]).tolist() == [0, 0, 1, 1, 2, 2]
    assert sev_from_tags([0, 1, 4, 5, 9, 10, 13]).tolist() == [-1, 0, 0, 1, 1, 2, 2]
    tags = np.arange(1, 14)
    assert (level_to_sev(tags) == sev_from_tags(tags)).all()


def test_mechanism_thresholds_at_the_edges():
    assert level_to_mech([1.0, 1.49, 1.5, 7.49, 7.5, 12]).tolist() == [0, 0, 1, 1, 2, 2]


def test_causal_median_uses_no_future_values():
    x = np.random.default_rng(0).normal(size=300)
    y = x.copy()
    y[200:] += 100
    assert np.allclose(causal_median(x)[:200], causal_median(y)[:200])


def test_tracking_never_decreases_after_start():
    lv = np.r_[np.ones(100), np.full(100, 8.0), np.full(100, 3.0)] + np.random.default_rng(1).normal(0, 0.3, 300)
    tr = track(lv, start=50)
    assert np.all(np.diff(tr[50:]) >= 0)
    assert tr[-1] > 7.5


def test_change_detector_alarms_on_a_step_and_not_on_noise():
    rng = np.random.default_rng(2)
    flat = 3 + rng.normal(0, 0.3, 2000)
    assert len(detect(flat)) == 0
    step = np.r_[3 + rng.normal(0, 0.3, 1000), 6 + rng.normal(0, 0.3, 1000)]
    al = detect(step)
    assert len(al) >= 1 and 1000 <= al[0] < 1200
