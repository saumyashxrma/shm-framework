"""REQ-F-06 environment compensation: temperature drift removed, 5 % false-alarm budget on the reference."""
import numpy as np

from shm.envcomp import EnvCompensator


def _healthy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    env = np.column_stack([rng.uniform(-10, 40, n), rng.uniform(20, 90, n)])
    level = 1.0 + 0.004 * (env[:, 0] - 15) ** 2 + rng.normal(0, 0.2, n)   # healthy level drifts with temperature
    return env, level


def test_false_alarm_budget_on_reference():
    env, lv = _healthy()
    ec = EnvCompensator.fit(env, lv)
    assert abs(ec.flag(env, lv).mean() - 0.05) < 0.01


def test_compensation_separates_cold_healthy_from_damage():
    env, lv = _healthy()
    ec = EnvCompensator.fit(env, lv)
    cold = np.array([[-8.0, 60.0]])
    healthy_cold = 1.0 + 0.004 * (-8 - 15) ** 2          # ~3.1: would look damaged without compensation
    assert not ec.flag(cold, [healthy_cold])[0]
    assert ec.flag(cold, [healthy_cold + 3.0])[0]
