"""Drive the stochastic-cloning EKF over a sequence with a learned displacement head."""

from dataclasses import dataclass

import numpy as np

from .ekf import EKFConfig, StochasticCloningEKF
from .geometry import so3_log


@dataclass
class RunConfig:
    window: int = 200          # samples per network input (1 s at 200 Hz)
    update_stride: int = 10    # samples between updates (20 Hz at 200 Hz)
    warmup: int = 200
    init_from_gt: bool = True
    overlap_inflation: bool = False  # scale R by window/stride when windows overlap


class FilterRun:
    """Container for filter output and per-update diagnostics."""

    def __init__(self):
        self.t = []
        self.p_est = []
        self.p_gt = []
        self.R_est = []
        self.R_gt = []
        self.v_est = []
        self.bg = []
        self.ba = []
        self.nis = []
        self.accepted = []
        self.sigma_pred = []
        self.disp_err = []
        self.pose_nees = []
        self.sigma_pose = []
        self.pose_err = []

    def finalize(self):
        for k, v in list(self.__dict__.items()):
            setattr(self, k, np.asarray(v))
        return self


def run_filter(seq, predict, run_cfg=None, ekf_cfg=None, progress=False):
    """Run the filter over `seq`.

    predict: callable taking (6, W) float32 array -> (disp(3,), var(3,))
    """
    rc = run_cfg or RunConfig()
    ekf_cfg = ekf_cfg or EKFConfig()
    ekf_cfg.max_clones = max(ekf_cfg.max_clones, rc.window // rc.update_stride + 2)
    if rc.overlap_inflation:
        ekf_cfg.cov_inflation = ekf_cfg.cov_inflation * rc.window / rc.update_stride
    f = StochasticCloningEKF(ekf_cfg)
    n = len(seq.ts)
    gyr_b, acc_b = seq.gyr_b(), seq.acc_b()

    if rc.init_from_gt:
        f.set_state(R=seq.R_wb[0], v=seq.v_w[0], p=seq.p_w[0])

    R_hist = np.empty((n, 3, 3))
    out = FilterRun()
    next_update = rc.warmup

    for i in range(n):
        if i > 0:
            f.propagate(gyr_b[i], acc_b[i], seq.ts[i] - seq.ts[i - 1])
        R_hist[i] = f.R

        if i % rc.update_stride == 0:
            f.clone(seq.ts[i])

        if i >= next_update and i >= rc.window and f.n_clones > 0:
            i0 = i - rc.window
            t0 = seq.ts[i0]
            k = int(np.argmin([abs(c.t - t0) for c in f.clones]))
            cl = f.clones[k]
            if abs(cl.t - t0) > 1e-6:
                next_update = i + rc.update_stride
                continue
            R_g = cl.R_g
            w = slice(i0, i)
            gyr_win = np.einsum("nij,nj->ni", R_hist[w], gyr_b[w] - f.bg) @ R_g
            acc_win = np.einsum("nij,nj->ni", R_hist[w], acc_b[w] - f.ba) @ R_g
            x = np.concatenate([gyr_win, acc_win], axis=1).T.astype(np.float32)

            if getattr(predict, "wants_context", False):
                disp, var = predict(x, {"i": i, "i0": i0, "R_g": R_g, "t": seq.ts[i]})
            else:
                disp, var = predict(x)
            info = f.update_displacement(disp, var, k=k)

            gt_disp = R_g.T @ (seq.p_w[i] - seq.p_w[i0])
            out.t.append(seq.ts[i])
            out.p_est.append(f.p.copy())
            out.p_gt.append(seq.p_w[i])
            out.R_est.append(f.R.copy())
            out.R_gt.append(seq.R_wb[i])
            out.v_est.append(f.v.copy())
            out.bg.append(f.bg.copy())
            out.ba.append(f.ba.copy())
            out.nis.append(info["nis"])
            out.accepted.append(info["accepted"])
            out.sigma_pred.append(np.sqrt(var))
            out.disp_err.append(disp - gt_disp)

            dtheta = so3_log(f.R.T @ seq.R_wb[i])
            dp = seq.p_w[i] - f.p
            e = np.concatenate([dtheta, dp])
            Ppose = f.pose_covariance()
            out.pose_nees.append(float(e @ np.linalg.solve(Ppose + 1e-12 * np.eye(6), e)))
            out.sigma_pose.append(np.sqrt(np.diag(Ppose)))
            out.pose_err.append(e)

            next_update = i + rc.update_stride

    return out.finalize()


def dead_reckon(seq):
    """IMU-only integration from ground-truth initial state, as a drift reference."""
    f = StochasticCloningEKF()
    f.set_state(R=seq.R_wb[0], v=seq.v_w[0], p=seq.p_w[0])
    gyr_b, acc_b = seq.gyr_b(), seq.acc_b()
    p = np.empty((len(seq.ts), 3))
    for i in range(len(seq.ts)):
        if i > 0:
            f.propagate(gyr_b[i], acc_b[i], seq.ts[i] - seq.ts[i - 1])
        p[i] = f.p
    return p
