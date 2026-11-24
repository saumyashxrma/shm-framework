"""O1 environment-compensated detection (K17).

A healthy plate's predicted damage level drifts with temperature and humidity (for example, healthy January 2019
recordings at about -11 deg C are predicted near level 6). The drift g(T, RH) is learned from HEALTHY reference
recordings only (quadratic ridge regression), subtracted, and the residual is compared with a threshold set at the
95th percentile of the reference residuals (5 % false-alarm budget).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from . import config as C


@dataclass
class EnvCompensator:
    model: object
    threshold: float

    @classmethod
    def fit(cls, env_healthy, level_healthy, quantile: float = C.ENV_FA_QUANTILE) -> "EnvCompensator":
        env_healthy = np.asarray(env_healthy, float)
        level_healthy = np.asarray(level_healthy, float)
        if env_healthy.ndim != 2 or env_healthy.shape[1] != 2:
            raise ValueError("env must be (n, 2): temperature (deg C), relative humidity (%)")
        g = make_pipeline(StandardScaler(), PolynomialFeatures(C.ENV_POLY_DEGREE), Ridge(alpha=C.ENV_RIDGE_ALPHA))
        g.fit(env_healthy, level_healthy)
        thr = float(np.quantile(level_healthy - g.predict(env_healthy), quantile))
        return cls(g, thr)

    def score(self, env, level) -> np.ndarray:
        """Environment-compensated damage score (level minus expected healthy level at this T, RH)."""
        return np.asarray(level, float) - self.model.predict(np.asarray(env, float))

    def flag(self, env, level) -> np.ndarray:
        """True = damage suspected."""
        return self.score(env, level) >= self.threshold
