"""End-to-end pipelines.

O1  recordings -> preprocessing -> CNN-BiLSTM level (3-seed ensemble) -> severity / mechanism
    level stream -> causal median -> cumulative tracking -> decision; change-detection alarms;
    environment-compensated detection score
O2  specimen history -> physics features -> GBM diagnosis (monotone) +/- q_d -> linear-trend + GP forecast +/- q_f
    -> inspection rule (flag when the 90 % upper bound reaches the critical crack length)

Run `python -m shm.pipeline --demo` for a console demonstration on the sample data (or synthetic data when the
exported sample files are not present).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import data as io
from .conformal import interval
from .envcomp import EnvCompensator
from .forecast import lin_gp_forecast
from .model_o2 import CrackDiagnoser, select_hp
from .tracking import detect, level_to_mech, level_to_sev, segment_starts, track


# ------------------------------------------------------------------ O1 ------------------------------------------------
@dataclass
class O1Decision:
    level: np.ndarray
    severity: np.ndarray
    mechanism: np.ndarray

    def summary(self, i: int = -1) -> str:
        return (f"level {self.level[i]:.2f} -> {C.SEV_NAMES[self.severity[i]]}, "
                f"mechanism {C.MECH_NAMES[self.mechanism[i]]}")


def decide(level) -> O1Decision:
    lv = np.asarray(level, float).ravel()
    return O1Decision(lv, level_to_sev(lv), level_to_mech(lv))


def assess_recordings(waves, env, fold: str = "A") -> O1Decision:
    """Per-recording assessment with the fold's 3-seed ensemble (needs TensorFlow and the exported models)."""
    from .model_o1 import LevelEnsemble
    return decide(LevelEnsemble(fold).predict(waves, env))


@dataclass
class O1StreamResult:
    tracked: O1Decision
    per_recording: O1Decision
    alarms: np.ndarray
    tracking_start: int


def monitor_stream(level, month=None, tracking_start: int | None = None) -> O1StreamResult:
    """Time-ordered level stream -> tracked decisions and change alarms (one baseline per campaign/month)."""
    lv = np.asarray(level, float)
    seg = segment_starts(month) if month is not None else None
    alarms = detect(lv, seg)
    start = int(tracking_start) if tracking_start is not None else (int(alarms[0]) if len(alarms) else len(lv))
    return O1StreamResult(decide(track(lv, start)), decide(lv), alarms, start)


# ------------------------------------------------------------------ O2 ------------------------------------------------
@dataclass
class O2Result:
    spec: str
    cycles: np.ndarray
    estimate: np.ndarray
    est_lo: np.ndarray
    est_hi: np.ndarray
    future_cycles: np.ndarray
    forecast: np.ndarray
    fc_lo: np.ndarray
    fc_hi: np.ndarray
    q_diag: float
    q_forecast: float
    inspect_now: bool = False
    first_flag_cycle: float | None = None
    notes: list = field(default_factory=list)


class O2Pipeline:
    def __init__(self, specimens: dict, info: dict | None = None):
        self.D = specimens
        self.info = info or io.load_o2_model_info()
        hp = self.info.get("hp") or select_hp(self.D)
        self.diagnoser = CrackDiagnoser(*hp).fit(self.D, C.TRAIN_SPECS)
        self.q_d = float(self.info.get("q_diag_mm", C.Q_DIAG_MM))
        self.q_f = float(self.info.get("q_forecast_mm", C.Q_FORECAST_MM))

    def run(self, spec: str, n_obs: int | None = None, future_cycles=None, a_crit: float = 5.0) -> O2Result:
        d = self.D[spec]
        n = len(d["cycles"]) if n_obs is None else int(np.clip(n_obs, 2, len(d["cycles"])))
        cyc, phys = d["cycles"][:n], d["phys"][:n]
        est = self.diagnoser.predict(cyc, phys)
        lo, hi = interval(est, self.q_d)
        if future_cycles is not None:
            fut = future_cycles
        elif n < len(d["cycles"]):
            fut = d["cycles"][n:]                      # the specimen's remaining inspections
        elif "future_cycles" in d:
            fut = d["future_cycles"]                   # challenge forecast points (T7/T8)
        else:
            fut = cyc[-1] + np.diff(cyc).mean() * np.arange(1, 6)
        fut = np.asarray(fut, float)
        fc = lin_gp_forecast(cyc, est, fut) if len(fut) else np.zeros(0)
        flo, fhi = interval(fc, self.q_f)
        flags = np.where(np.r_[hi, fhi] >= a_crit)[0]
        all_c = np.r_[cyc, fut]
        return O2Result(spec, cyc, est, lo, hi, fut, fc, flo, fhi, self.q_d, self.q_f,
                        inspect_now=bool(hi[-1] >= a_crit),
                        first_flag_cycle=float(all_c[flags[0]]) if len(flags) else None)


# ------------------------------------------------------------------ synthetic data (CI / no sample files) ------------
def synthetic_specimens(seed: int = 0) -> dict:
    """Small synthetic stand-in with the same structure as the exported O2 sample (for tests and offline demo)."""
    rng = np.random.default_rng(seed)
    D = {}
    for k, n in enumerate(C.TRAIN_SPECS + C.CHALLENGE_SPECS):
        cyc = np.linspace(0, 60000 + 5000 * k, 9)
        crack = np.clip((cyc - 20000) / 8000, 0, None) + rng.normal(0, 0.1, len(cyc)).clip(0)
        phys = np.column_stack([-0.05 * crack + rng.normal(0, 0.02, len(cyc)) for _ in range(len(C.PHYS_NAMES))])
        phys[0] = 0.0
        D[n] = {"cycles": cyc, "cracks": crack, "phys": phys}
    return D


def synthetic_stream(seed: int = 0):
    rng = np.random.default_rng(seed)
    tags = np.r_[np.zeros(600), np.full(600, 3), np.full(600, 7), np.full(600, 11)]
    level = tags + rng.normal(0, 0.5, len(tags))
    month = np.repeat(np.array(["m1", "m2"]), 1200)            # two campaigns; damage changes inside them
    return level, tags, month


def synthetic_o1_recordings(seed: int = 0) -> dict:
    """SYNTHETIC stand-in for data/sample/o1_recordings.npz (same keys). For demonstrating the interface only."""
    rng = np.random.default_rng(seed)
    tags = np.repeat(np.arange(14), 2)
    t = np.arange(C.N_SAMP) / (C.FS_UTAH / C.STRIDE)
    waves = np.zeros((len(tags), C.N_SAMP, C.N_CH), np.float32)
    for i, tag in enumerate(tags):
        for ch in range(C.N_CH):
            t0 = 1e-4 + 2e-5 * ch
            burst = np.exp(-((t - t0) / 3e-5) ** 2) * np.sin(2 * np.pi * 100e3 * (t - t0))
            scat = 0.04 * tag * np.exp(-((t - t0 - 2e-4) / 6e-5) ** 2) * np.sin(2 * np.pi * 120e3 * t)
            w = burst + scat + 0.02 * rng.standard_normal(len(t))
            waves[i, :, ch] = w / np.abs(w).max()
    env = np.c_[rng.uniform(-5, 35, len(tags)), rng.uniform(20, 90, len(tags))]
    out = {"waves": waves, "env": env, "tags": tags, "month": np.array(["synthetic"] * len(tags))}
    for f, hid in C.TAG_FOLDS.items():                  # unseen states read low, as in the real folds
        out[f"ref_level_{f}"] = tags + rng.normal(0, 0.4, len(tags)) - np.isin(tags, hid) * 1.2
    return out


def synthetic_o1_stream_full(seed: int = 0) -> dict:
    """SYNTHETIC stand-in for data/sample/o1_stream.npz (same keys). For demonstrating the interface only."""
    rng = np.random.default_rng(seed)
    level, tags, month = synthetic_stream(seed)
    env = np.c_[15 + 10 * np.sin(np.linspace(0, 6, len(tags))) + rng.normal(0, 1, len(tags)), rng.uniform(30, 90, len(tags))]
    out = {"tags": tags, "month": month, "env": env}
    for f, hid in C.TAG_FOLDS.items():
        out[f"level_{f}"] = tags + rng.normal(0, 0.5, len(tags)) - np.isin(tags, hid) * 1.2 + 0.03 * (env[:, 0] - 15)
    return out


# ------------------------------------------------------------------ CLI ----------------------------------------------
def _demo() -> None:
    print("SHM framework v1.0.0 - console demonstration\n")
    if io.has(io.O1_STREAM):
        s = io.load_o1_stream()
        lv, tags, month = s["level_A"], s["tags"], s["month"]
        start = int(np.argmax(tags >= 2))
        src = "exported fold-A level stream (Kaggle run)"
    else:
        lv, tags, month = synthetic_stream()
        start, src = 600, "SYNTHETIC stream (export data/sample/o1_stream.npz for the real one)"
    r = monitor_stream(lv, month, tracking_start=start)
    print(f"O1 stream: {src}")
    print(f"   {len(lv)} recordings | {len(r.alarms)} change alarms | tracking from recording {r.tracking_start}")
    print(f"   last recording: per-recording {r.per_recording.summary()} | tracked {r.tracked.summary()}\n")

    if io.has(io.O2_SPECIMENS):
        D, src = io.load_o2_specimens(), "exported NASA/PHM-2019 features (Colab run)"
    else:
        D, src = synthetic_specimens(), "SYNTHETIC specimens (export data/sample/o2_specimens.json for the real ones)"
    p = O2Pipeline(D)
    print(f"O2: {src} | diagnoser hp {p.diagnoser.hp} | +/-{p.q_d:.2f} mm diagnosis, +/-{p.q_f:.2f} mm forecast")
    for spec in C.CHALLENGE_SPECS:
        o = p.run(spec)
        print(f"   {spec}: latest estimate {o.estimate[-1]:.2f} mm [{o.est_lo[-1]:.2f}, {o.est_hi[-1]:.2f}] at "
              f"{o.cycles[-1]:.0f} cycles | forecast at {o.future_cycles[-1]:.0f} cycles: {o.forecast[-1]:.2f} mm "
              f"[{o.fc_lo[-1]:.2f}, {o.fc_hi[-1]:.2f}] | inspect now (5 mm limit): {o.inspect_now}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SHM framework pipelines")
    ap.add_argument("--demo", action="store_true", help="run the console demonstration")
    args = ap.parse_args(argv)
    if args.demo:
        _demo()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()