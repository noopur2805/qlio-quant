"""Numerical parity check: qlio.ekf.StochasticCloningEKF (Python) vs qlio_ekf (C++).

Run from the repo root:
    PYTHONPATH=. python cpp/parity_check.py

Drives both implementations through an identical randomized 2000-step schedule
(propagation, cloning/marginalisation, gated displacement updates including
outliers, batch updates) for both use_fej=True and use_fej=False, comparing
R, v, p, bg, ba, every clone's R/p, and P after every step.
"""

import sys

import numpy as np

from qlio.ekf import EKFConfig, StochasticCloningEKF
import qlio_ekf  # found via the script's own directory (cpp/)

ATOL, RTOL = 1e-10, 1e-8
STEPS = 2000
DT = 0.005


def compare(step, tag, a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.allclose(a, b, atol=ATOL, rtol=RTOL):
        diff = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("nan")
        print(f"FAIL at step {step} [{tag}]: max abs diff = {diff:.6e}")
        sys.exit(1)


def check_states(step, f_py, f_cpp):
    if f_py.n_clones != f_cpp.n_clones or f_py.dim != f_cpp.dim:
        print(f"FAIL at step {step}: n_clones/dim mismatch "
              f"({f_py.n_clones}/{f_py.dim} vs {f_cpp.n_clones}/{f_cpp.dim})")
        sys.exit(1)
    for tag in ("R", "v", "p", "bg", "ba", "P"):
        compare(step, tag, getattr(f_py, tag), getattr(f_cpp, tag))
    compare(step, "pose_covariance", f_py.pose_covariance(), f_cpp.pose_covariance())
    for k, cl in enumerate(f_py.clones):
        c = f_cpp.get_clone(k)
        if cl.cid != c["cid"]:
            print(f"FAIL at step {step}: clone {k} cid mismatch ({cl.cid} vs {c['cid']})")
            sys.exit(1)
        compare(step, f"clone[{k}].R", cl.R, c["R"])
        compare(step, f"clone[{k}].p", cl.p, c["p"])
        compare(step, f"clone[{k}].R_g", cl.R_g, c["R_g"])


def run_variant(use_fej, chi2_gate, seed, steps=STEPS, label=""):
    rng = np.random.default_rng(seed)
    cfg = EKFConfig(max_clones=22, chi2_gate=chi2_gate,
                    cov_inflation=1.3, use_fej=use_fej)
    f_py = StochasticCloningEKF(cfg)
    f_cpp = qlio_ekf.StochasticCloningEKF(max_clones=22, chi2_gate=chi2_gate,
                                          cov_inflation=1.3, use_fej=use_fej)
    n_gated = 0
    t = 0.0
    for step in range(1, steps + 1):
        gyr = rng.normal(0.0, 0.4, 3)
        acc = np.array([0.0, 0.0, 9.81]) + rng.normal(0.0, 0.6, 3)
        f_py.propagate(gyr, acc, DT)
        f_cpp.propagate(gyr, acc, DT)
        t += DT

        if step % 10 == 0:
            cid_py = f_py.clone(t)
            cid_cpp = f_cpp.clone(t)
            if cid_py != cid_cpp or f_py.clone_index(cid_py) != f_cpp.clone_index(cid_cpp):
                print(f"FAIL at step {step}: cid/clone_index mismatch")
                sys.exit(1)

            # displacement update on a random live clone
            k = int(rng.integers(0, f_py.n_clones))
            _, h = f_py.displacement_jacobian(k)
            cov = rng.uniform(1e-4, 1e-2, 3)  # random diagonal covariance
            if rng.random() < 0.1:
                z = h + 100.0 + rng.normal(0.0, 50.0, 3)  # huge outlier -> chi2 gate
            else:
                z = h + rng.normal(0.0, 1.0, 3) * np.sqrt(cov)
            res_py = f_py.update_displacement(z, cov, k)
            res_cpp = f_cpp.update_displacement(z, cov, k)
            if res_py["accepted"] != res_cpp["accepted"]:
                print(f"FAIL at step {step}: accepted mismatch "
                      f"({res_py['accepted']} vs {res_cpp['accepted']}, "
                      f"nis {res_py['nis']:.6g} vs {res_cpp['nis']:.6g})")
                sys.exit(1)
            compare(step, "nis", res_py["nis"], res_cpp["nis"])
            compare(step, "residual", res_py["residual"], res_cpp["residual"])
            if not res_py["accepted"]:
                n_gated += 1

        if step % 50 == 0:
            d = f_py.dim
            H = rng.normal(0.0, 0.1, (6, d))
            r = rng.normal(0.0, 0.05, 6)
            Rm = np.diag(rng.uniform(1e-3, 1e-2, 6))
            b_py = f_py.update_batch(H, r, Rm)
            b_cpp = f_cpp.update_batch(H, r, Rm)
            compare(step, "dx_norm", b_py["dx_norm"], b_cpp["dx_norm"])

        check_states(step, f_py, f_cpp)

    gate_txt = "None" if chi2_gate is None else f"{chi2_gate:g}"
    print(f"  variant {label} (use_fej={use_fej}, chi2_gate={gate_txt}): "
          f"{steps} steps OK, {n_gated} updates gated")
    if chi2_gate is not None and n_gated == 0:
        print("  WARNING: chi2 gate was never exercised")
        sys.exit(1)


def main():
    run_variant(use_fej=True, chi2_gate=30.0, seed=42, label="fej")
    run_variant(use_fej=False, chi2_gate=30.0, seed=43, label="legacy")
    # short run with the gate disabled (chi2_gate=None must be accepted)
    run_variant(use_fej=True, chi2_gate=None, seed=44, steps=300, label="no-gate")
    print("PASS")


if __name__ == "__main__":
    main()
