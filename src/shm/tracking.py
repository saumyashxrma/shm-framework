"""O1 decision layer: damage level -> severity / mechanism, cumulative tracking and change detection (K16).

The Stage-2 network outputs a continuous damage level on the tag scale (0-13). Fixed thresholds turn it into
decisions; nothing here is fitted to test data.

Cumulative tracking (adopted by the pre-declared K16 rule): damage does not heal, so once permanent damage is
confirmed the reported level is the running maximum of a causal rolling median. In the experiments tracking starts
at the first tag >= 2 recording; in operation that start comes from a confirmed alarm or an inspection
(stated as an assumption in the report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def level_to_sev(level) -> np.ndarray:
    """0 = Monitor (< 4.5), 1 = Inspect (4.5 to < 9.5), 2 = Emergency (>= 9.5)."""
    return np.digitize(np.asarray(level, float).ravel(), C.SEV_THR).astype(np.int32)


def level_to_mech(level) -> np.ndarray:
    """0 = mass (< 1.5), 1 = dent (1.5 to < 7.5), 2 = through-hole (>= 7.5)."""
    return np.digitize(np.asarray(level, float).ravel(), C.MECH_THR).astype(np.int32)


def sev_from_tags(tags) -> np.ndarray:
    """Ground-truth severity from damage tags: 1-4 Monitor, 5-9 Inspect, 10-13 Emergency, 0 -> -1 (healthy)."""
    t = np.asarray(tags)
    s = np.full(len(t), -1, np.int32)
    s[(t >= 1) & (t <= 4)] = 0
    s[(t >= 5) & (t <= 9)] = 1
    s[t >= 10] = 2
    return s


def causal_median(x, w: int = C.W_MED) -> np.ndarray:
    """Rolling median over the current and previous w-1 values (uses no future recordings)."""
    return pd.Series(np.asarray(x, float)).rolling(w, min_periods=1).median().to_numpy()


def track(level, start: int, w: int = C.W_MED) -> np.ndarray:
    """Causal median, then running maximum from index `start` (first confirmed permanent damage) onwards."""
    med = causal_median(level, w)
    out = med.copy()
    if 0 <= start < len(out):
        out[start:] = np.maximum.accumulate(med[start:])
    return out


def detect(level, segment_starts=None, delta: float = C.DELTA, w: int = C.W_MED,
           b_ref: int = C.B_REF, persist: int = C.PERSIST) -> np.ndarray:
    """Causal change detector. Each segment (measurement campaign) builds its own baseline from its first b_ref
    recordings; an alarm is raised when the smoothed level stays more than `delta` away from the baseline for
    `persist` consecutive recordings, after which the baseline is rebuilt. Returns alarm indices."""
    lv = np.asarray(level, float)
    starts = [0] if segment_starts is None else sorted(set(int(s) for s in segment_starts) | {0})
    bounds = starts + [len(lv)]
    alarms = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = causal_median(lv[a:b], w)
        base, start, cnt = None, 0, 0
        for i in range(len(m)):
            if base is None:
                if i - start + 1 >= b_ref:
                    base = float(np.median(lv[a + start:a + i + 1]))
                continue
            cnt = cnt + 1 if abs(m[i] - base) > delta else 0
            if cnt >= persist:
                alarms.append(a + i)
                base, start, cnt = None, i + 1, 0
    return np.array(alarms, int)


def segment_starts(month) -> np.ndarray:
    """Indices where the campaign label changes (start of each month in a time-ordered stream)."""
    m = np.asarray(month)
    return np.r_[0, np.where(m[1:] != m[:-1])[0] + 1]
