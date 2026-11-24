"""O2 physics features for lap-joint crack diagnosis (identical to notebook cell Q1, fixed version).

Each recording is compared with the SAME specimen's first recording (its own healthy reference), after dividing by
the excitation peak. Nine features describe how a crack changes the received Lamb wave: amplitude and energy loss,
phase lag, decorrelation, energy in the first packet versus everything outside it (scattered/reflected waves),
envelope-peak shift and spectral-centroid shift.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.signal import butter, correlate, filtfilt, hilbert

from . import config as C

EPS = 1e-12
_BQ, _AQ = butter(4, [C.NASA_BAND[0] / (C.FS_NASA / 2), C.NASA_BAND[1] / (C.FS_NASA / 2)], btype="band")


def bandpass_nasa(x) -> np.ndarray:
    return filtfilt(_BQ, _AQ, np.asarray(x, float))


def phys_features(rx, tx_peak) -> np.ndarray:
    """rx: (n_recordings, L) filtered receiver signals in time order (row 0 = reference);
    tx_peak: (n_recordings,) excitation peak amplitudes. Returns (n_recordings, 9)."""
    rx = np.asarray(rx, float)
    tx_peak = np.asarray(tx_peak, float)
    if rx.ndim != 2 or len(rx) != len(tx_peak):
        raise ValueError("rx must be (n, L) and tx_peak (n,)")
    s0 = rx[0] / tx_peak[0]
    e0 = np.abs(hilbert(s0))
    k0 = int(np.argmax(e0))
    L = len(s0)
    lo, hi = max(0, k0 - C.PACKET_HALF_WIDTH), min(L, k0 + C.PACKET_HALF_WIDTH)
    pk = np.zeros(L, bool)
    pk[lo:hi] = True
    late = ~pk                                         # everything outside the first packet
    z0 = (s0 - s0.mean()) / s0.std()
    sp0 = np.abs(np.fft.rfft(s0))
    f = np.arange(len(sp0))
    c0 = (sp0 * f).sum() / sp0.sum()
    feats = []
    for s_raw, tp in zip(rx, tx_peak):
        s = s_raw / tp
        e = np.abs(hilbert(s))
        z = (s - s.mean()) / s.std()
        cc = correlate(z, z0, mode="full") / len(z)
        mid = len(z) - 1
        w = cc[mid - C.MAXLAG: mid + C.MAXLAG + 1]
        lag = int(np.argmax(np.abs(w)) - C.MAXLAG)
        sp = np.abs(np.fft.rfft(s))
        feats.append([
            np.log((e.max() + EPS) / (e0.max() + EPS)),
            np.log(((s ** 2).sum() + EPS) / ((s0 ** 2).sum() + EPS)),
            lag,
            1.0 - np.max(np.abs(w)),
            1.0 - np.corrcoef(s, s0)[0, 1],
            np.log(((e[pk] ** 2).sum() + EPS) / ((e0[pk] ** 2).sum() + EPS)),
            np.log(((e[late] ** 2).sum() + EPS) / ((e0[late] ** 2).sum() + EPS)),
            int(np.argmax(e)) - k0,
            (sp * f).sum() / sp.sum() / c0 - 1.0,
        ])
    return np.nan_to_num(np.array(feats, float), nan=0.0, posinf=0.0, neginf=0.0)


def load_nasa_specimen(spec_dir: str, labels: dict):
    """Read one specimen folder of the PHM 2019 data (sub-folders named by cycle count, each with signal_1.csv /
    signal_2.csv holding ch1 = excitation, ch2 = response). labels: {cycle: crack length mm}.
    Returns dict(cycles, cracks, rx, tx_peak) sorted by cycles."""
    import pandas as pd
    rec = []
    for d in os.listdir(spec_dir):
        p = os.path.join(spec_dir, d)
        if not (os.path.isdir(p) and d.isdigit() and int(d) in labels):
            continue
        rx, tx = [], []
        for r in ("signal_1.csv", "signal_2.csv"):
            fp = os.path.join(p, r)
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                rx.append(bandpass_nasa(df["ch2"].values))
                tx.append(np.abs(df["ch1"].values.astype(float)).max())
        if rx:
            rec.append((int(d), labels[int(d)], np.mean(rx, 0), float(np.mean(tx))))
    rec.sort(key=lambda r: r[0])
    return {"cycles": np.array([r[0] for r in rec], float), "cracks": np.array([r[1] for r in rec], float),
            "rx": np.stack([r[2] for r in rec]), "tx_peak": np.array([r[3] for r in rec])}
