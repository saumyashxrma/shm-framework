"""O2 crack-length diagnosis: gradient-boosted trees on [load cycles, 9 physics features] + causal monotone constraint.

Hyper-parameters (depth, number of trees) are chosen by leave-one-specimen-out on the training specimens only
(T1-T6); the evaluation in the report nests this selection inside an outer leave-one-specimen-out loop.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from . import config as C


def causal_monotone(pred) -> np.ndarray:
    """A crack cannot shrink: running maximum over the estimates so far (uses no future data)."""
    return np.maximum.accumulate(np.asarray(pred, float))


def _xy(D, specs):
    X = np.vstack([np.hstack([D[n]["cycles"][:, None], D[n]["phys"]]) for n in specs])
    y = np.concatenate([D[n]["cracks"] for n in specs])
    return X, y


class CrackDiagnoser:
    def __init__(self, depth: int = 2, n_estimators: int = 150, seed: int = 0):
        self.hp = (int(depth), int(n_estimators))
        self.model = GradientBoostingRegressor(max_depth=depth, n_estimators=n_estimators, learning_rate=0.05,
                                               subsample=0.8, min_samples_leaf=3, random_state=seed)

    def fit(self, D, specs) -> "CrackDiagnoser":
        X, y = _xy(D, specs)
        self.model.fit(X, y)
        return self

    def predict_raw(self, cycles, phys) -> np.ndarray:
        X = np.hstack([np.asarray(cycles, float)[:, None], np.asarray(phys, float)])
        return np.clip(self.model.predict(X), 0, None)

    def predict(self, cycles, phys) -> np.ndarray:
        """Monotone (causal) crack-length estimates in mm for a specimen's history in time order."""
        return causal_monotone(self.predict_raw(cycles, phys))


def select_hp(D, specs=C.TRAIN_SPECS, grid=C.GB_GRID, seed: int = 0):
    """Leave-one-specimen-out MAE over `specs` for every grid point; returns the best (depth, n_estimators)."""
    def score(hp):
        errs = []
        for v in specs:
            m = CrackDiagnoser(*hp, seed=seed).fit(D, [s for s in specs if s != v])
            errs.append(mean_absolute_error(D[v]["cracks"], m.predict(D[v]["cycles"], D[v]["phys"])))
        return float(np.mean(errs))
    return min(grid, key=score)
