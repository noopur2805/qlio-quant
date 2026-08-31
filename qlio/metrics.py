"""Trajectory error, filter consistency, and uncertainty calibration metrics."""

import numpy as np
from scipy.stats import chi2


def ate(p_est, p_gt):
    """Absolute translation error (RMSE), no alignment."""
    return float(np.sqrt(np.mean(np.sum((p_est - p_gt) ** 2, axis=1))))


def rte(t, p_est, p_gt, horizon=10.0):
    """Relative translation error over a fixed time horizon."""
    errs = []
    j = 0
    for i in range(len(t)):
        while j < len(t) and t[j] - t[i] < horizon:
            j += 1
        if j >= len(t):
            break
        d_est = p_est[j] - p_est[i]
        d_gt = p_gt[j] - p_gt[i]
        errs.append(np.linalg.norm(d_est - d_gt))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def drift_ratio(p_est, p_gt):
    dist = float(np.sum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1)))
    return float(np.linalg.norm(p_est[-1] - p_gt[-1]) / max(dist, 1e-9))


def nees_stats(nees, dof=6, alpha=0.05):
    """Average NEES with the two-sided chi-square acceptance interval."""
    nees = np.asarray(nees, dtype=float)
    n = len(nees)
    avg = float(np.mean(nees))
    lo = chi2.ppf(alpha / 2, dof * n) / n
    hi = chi2.ppf(1 - alpha / 2, dof * n) / n
    return {
        "avg_nees": avg,
        "normalized": avg / dof,
        "bounds": (float(lo), float(hi)),
        "consistent": bool(lo <= avg <= hi),
        "n": n,
    }


def nis_stats(nis, dof=3, alpha=0.05):
    out = nees_stats(nis, dof=dof, alpha=alpha)
    out["avg_nis"] = out.pop("avg_nees")
    return out


def sigma_coverage(errors, sigmas, ks=(1.0, 2.0, 3.0)):
    """Fraction of per-axis errors inside k sigma, versus the Gaussian ideal."""
    errors = np.asarray(errors)
    sigmas = np.asarray(sigmas)
    z = np.abs(errors) / np.maximum(sigmas, 1e-12)
    ideal = {1.0: 0.6827, 2.0: 0.9545, 3.0: 0.9973}
    return {
        f"within_{k:g}sigma": {
            "observed": float(np.mean(z <= k)),
            "ideal": ideal.get(k, float("nan")),
        }
        for k in ks
    }


def gaussian_nll(errors, sigmas):
    errors = np.asarray(errors)
    var = np.maximum(np.asarray(sigmas) ** 2, 1e-24)
    return float(np.mean(0.5 * (errors**2 / var + np.log(2 * np.pi * var))))


def overconfidence(errors, sigmas):
    """Mean squared z-score; 1.0 is calibrated, >1 is over-confident."""
    z = np.asarray(errors) / np.maximum(np.asarray(sigmas), 1e-12)
    return float(np.mean(z**2))


def fit_variance_scale(errors, sigmas, per_axis=True):
    """Closed-form NLL-optimal multiplier on sigma (temperature scaling)."""
    z2 = (np.asarray(errors) / np.maximum(np.asarray(sigmas), 1e-12)) ** 2
    if per_axis:
        return np.sqrt(np.mean(z2, axis=0))
    return np.full(z2.shape[1], np.sqrt(np.mean(z2)))


def displacement_metrics(errors, sigmas):
    errors = np.asarray(errors)
    return {
        "rmse_m": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        "mae_m": float(np.mean(np.abs(errors))),
        "nll": gaussian_nll(errors, sigmas),
        "mean_z2": overconfidence(errors, sigmas),
        "mean_sigma_m": float(np.mean(sigmas)),
        **sigma_coverage(errors, sigmas),
    }
