"""Benchmark: Python vs C++ StochasticCloningEKF on an identical schedule.

Run from the repo root:
    PYTHONPATH=. python cpp/bench.py

Schedule: 20k propagate steps (dt=0.005), clone + gated displacement update on
a random live clone every 10 steps (max_clones=22, so dim reaches 147 and
marginalisation is exercised), batch update with a (6, dim) H every 50 steps.
"""

import time

import numpy as np

from qlio.ekf import EKFConfig, StochasticCloningEKF
import qlio_ekf

STEPS = 20000
DT = 0.005
MAX_CLONES = 22


def make_inputs(seed):
    rng = np.random.default_rng(seed)
    gyr = rng.normal(0.0, 0.4, (STEPS, 3))
    acc = np.array([0.0, 0.0, 9.81]) + rng.normal(0.0, 0.6, (STEPS, 3))
    kdraw = rng.integers(0, 1 << 30, STEPS)
    znoise = rng.normal(0.0, 1.0, (STEPS, 3))
    cov = rng.uniform(1e-4, 1e-2, (STEPS, 3))
    H6 = rng.normal(0.0, 0.1, (STEPS, 6, 15 + 6 * MAX_CLONES))
    r6 = rng.normal(0.0, 0.05, (STEPS, 6))
    R6 = rng.uniform(1e-3, 1e-2, (STEPS, 6))
    return gyr, acc, kdraw, znoise, cov, H6, r6, R6


def run(f, inputs):
    gyr, acc, kdraw, znoise, cov, H6, r6, R6 = inputs
    t = 0.0
    t0 = time.perf_counter()
    for i in range(STEPS):
        f.propagate(gyr[i], acc[i], DT)
        t += DT
        step = i + 1
        if step % 10 == 0:
            f.clone(t)
            k = int(kdraw[i] % f.n_clones)
            _, h = f.displacement_jacobian(k)
            z = h + znoise[i] * np.sqrt(cov[i])
            f.update_displacement(z, cov[i], k)
        if step % 50 == 0:
            f.update_batch(H6[i][:, :f.dim], r6[i], np.diag(R6[i]))
    return time.perf_counter() - t0


def main():
    inputs = make_inputs(2024)
    cfg = EKFConfig(max_clones=MAX_CLONES, chi2_gate=30.0, use_fej=True)
    f_py = StochasticCloningEKF(cfg)
    f_cpp = qlio_ekf.StochasticCloningEKF(max_clones=MAX_CLONES, chi2_gate=30.0,
                                          use_fej=True)

    t_cpp = run(f_cpp, inputs)   # C++ first (warm caches favour neither much)
    t_py = run(f_py, inputs)

    print(f"schedule: {STEPS} propagate steps, clone+update every 10, "
          f"batch update every 50, max_clones={MAX_CLONES} (dim up to {15 + 6 * MAX_CLONES})")
    print(f"python : {t_py:8.3f} s  ({STEPS / t_py:10.1f} steps/s)")
    print(f"c++    : {t_cpp:8.3f} s  ({STEPS / t_cpp:10.1f} steps/s)")
    print(f"speedup: {t_py / t_cpp:.1f}x")


if __name__ == "__main__":
    main()
