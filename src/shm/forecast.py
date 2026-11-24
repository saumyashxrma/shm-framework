"""O2 crack-growth forecasting: linear trend + Gaussian-process residual (Matern nu = 1.5), as in notebook cell Q3.

Selected for robustness: lowest mean (1.15 mm) and median RMSE and the smallest worst case (1.89 mm) over
T1-T6 from true histories; the similarity ensemble and Paris-law fits were better on some specimens but failed
badly on others (see the report, Chapter 7).
"""
from __future__ import annotations

import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def lin_gp_forecast(cycles_hist, crack_hist, cycles_future) -> np.ndarray:
    ch, ah, cf = (np.asarray(v, float) for v in (cycles_hist, crack_hist, cycles_future))
    if len(ch) < 2:
        raise ValueError("at least two past inspections are needed to forecast")
    mu, sd = ch.mean(), ch.std() + 1e-6
    slope, icpt = np.polyfit(ch, ah, 1)
    k = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
         + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0)))
    gp = GaussianProcessRegressor(kernel=k, alpha=1e-6, n_restarts_optimizer=3, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(((ch - mu) / sd).reshape(-1, 1), ah - (slope * ch + icpt))
    return np.maximum(slope * cf + icpt + gp.predict(((cf - mu) / sd).reshape(-1, 1)), 0)
