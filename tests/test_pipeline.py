"""End-to-end smoke tests (synthetic data) and checks against the exported sample data when present."""
import numpy as np
import pytest

from shm import config as C
from shm import data as io
from shm.pipeline import O2Pipeline, decide, main, monitor_stream, synthetic_specimens, synthetic_stream


def test_o1_stream_end_to_end():
    lv, tags, month = synthetic_stream()
    r = monitor_stream(lv, month, tracking_start=600)
    assert len(r.tracked.level) == len(lv)
    assert r.tracked.severity[-1] == 2 and r.tracked.mechanism[-1] == 2
    assert np.all(np.diff(r.tracked.level[600:]) >= 0)


def test_o2_end_to_end_with_inspection_rule():
    p = O2Pipeline(synthetic_specimens(), info={"hp": [1, 50], "q_diag_mm": 3.08, "q_forecast_mm": 4.73})
    o = p.run("T7", n_obs=6, a_crit=2.0)
    assert len(o.forecast) == 3 and np.allclose(o.est_hi - o.estimate, 3.08)
    assert o.inspect_now == bool(o.est_hi[-1] >= 2.0)


def test_cli_demo_runs(capsys):
    main(["--demo"])
    assert "O1 stream" in capsys.readouterr().out


def test_decide_names():
    d = decide([0.5, 6.0, 12.0])
    assert [C.SEV_NAMES[i] for i in d.severity] == ["Monitor", "Inspect", "Emergency"]


@pytest.mark.skipif(not io.has(io.O2_SPECIMENS), reason="exported O2 sample not present")
def test_exported_o2_reproduces_colab_reference():
    info = io.load_o2_model_info()
    p = O2Pipeline(io.load_o2_specimens(), info)
    for spec, ref in info.get("reference", {}).items():
        est = p.diagnoser.predict(p.D[spec]["cycles"], p.D[spec]["phys"])
        assert np.allclose(est, ref["estimate"], atol=0.05), spec


@pytest.mark.skipif(not io.has(io.O1_RECORDINGS), reason="exported O1 sample not present")
def test_exported_o1_models_reproduce_kaggle_levels():
    from shm import model_o1
    if not model_o1.available("A"):
        pytest.skip("TensorFlow or fold-A models not available")
    r = io.load_o1_recordings()
    lv = model_o1.LevelEnsemble("A").predict(r["waves"].astype(np.float32), r["env"])
    assert np.allclose(lv, r["ref_level_A"], atol=0.05)
