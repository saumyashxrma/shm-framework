"""Split-conformal prediction intervals (absolute-residual score).

q is the ceil((n + 1) * level)-th smallest absolute calibration residual; the interval is estimate +/- q.
O2 release values (deployment-matched calibration on T1-T6): diagnosis +/- 3.08 mm, forecast +/- 4.73 mm,
88.9 % (16/18) coverage on the held-out challenge specimens T7/T8.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def conformal_q(residuals, level: float = C.CONFORMAL_LEVEL) -> float:
    res = np.sort(np.abs(np.asarray(residuals, float)))
    n = len(res)
    if n == 0:
        raise ValueError("no calibration residuals")
    return float(res[min(int(np.ceil((n + 1) * level)), n) - 1])


def interval(estimate, q: float, lower_bound: float = 0.0):
    est = np.asarray(estimate, float)
    return np.maximum(est - q, lower_bound), est + q


def coverage(truth, estimate, q: float) -> float:
    return float(np.mean(np.abs(np.asarray(truth, float) - np.asarray(estimate, float)) <= q))
