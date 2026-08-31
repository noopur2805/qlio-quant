"""Stochastic-cloning EKF fusing IMU propagation with learned displacement updates.

Error state: [dtheta, dv, dp, dbg, dba] (15) followed by 6 per clone
[dtheta_c, dp_c]. Attitude error is right-invariant, R = R_hat exp(dtheta).
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import (gravity_aligned_frame, skew, so3_exp,
                       so3_right_jacobian_inv, yaw_world_jacobian)

GRAVITY = np.array([0.0, 0.0, -9.81])
E3 = np.array([0.0, 0.0, 1.0])


@dataclass
class EKFConfig:
    sigma_g: float = 2e-3          # gyro white noise [rad/s/sqrt(Hz)]
    sigma_a: float = 2e-2          # accel white noise [m/s^2/sqrt(Hz)]
    sigma_bg: float = 1e-4         # gyro bias random walk
    sigma_ba: float = 1e-3         # accel bias random walk
    init_sigma_theta: float = 1e-2
    init_sigma_v: float = 1e-1
    init_sigma_p: float = 1e-3
    init_sigma_bg: float = 5e-3
    init_sigma_ba: float = 3e-2
    max_clones: int = 2
    chi2_gate: float = 30.0        # 3-DoF gate; None disables
    cov_inflation: float = 1.0     # scales the network covariance fed to the filter


@dataclass
class Clone:
    R: np.ndarray
    p: np.ndarray
    t: float
    cid: int = -1
    R_g: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.R_g is None:
            self.R_g = gravity_aligned_frame(self.R)


class StochasticCloningEKF:
    def __init__(self, cfg=None):
        self.cfg = cfg or EKFConfig()
        self.R = np.eye(3)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.clones = deque()
        self._next_cid = 0
        c = self.cfg
        self.P = np.diag(np.concatenate([
            np.full(3, c.init_sigma_theta**2),
            np.full(3, c.init_sigma_v**2),
            np.full(3, c.init_sigma_p**2),
            np.full(3, c.init_sigma_bg**2),
            np.full(3, c.init_sigma_ba**2),
        ]))

    # ---- state layout -------------------------------------------------
    @property
    def n_clones(self):
        return len(self.clones)

    @property
    def dim(self):
        return 15 + 6 * self.n_clones

    def set_state(self, R=None, v=None, p=None, bg=None, ba=None):
        for name, val in (("R", R), ("v", v), ("p", p), ("bg", bg), ("ba", ba)):
            if val is not None:
                setattr(self, name, np.array(val, dtype=np.float64))

    # ---- propagation ---------------------------------------------------
    def propagate(self, gyr_b, acc_b, dt):
        c = self.cfg
        w = gyr_b - self.bg
        a = acc_b - self.ba
        Ra = self.R @ a

        dR = so3_exp(w * dt)
        self.p = self.p + self.v * dt + 0.5 * (Ra + GRAVITY) * dt**2
        self.v = self.v + (Ra + GRAVITY) * dt
        self.R = self.R @ dR

        F = np.eye(15)
        F[0:3, 0:3] = dR.T
        F[0:3, 9:12] = -so3_right_jacobian_inv(w * dt).T * dt
        F[3:6, 0:3] = -self.R @ skew(a) * dt
        F[3:6, 12:15] = -self.R * dt
        F[6:9, 3:6] = np.eye(3) * dt
        F[6:9, 0:3] = -0.5 * self.R @ skew(a) * dt**2
        F[6:9, 12:15] = -0.5 * self.R * dt**2

        Qd = np.zeros((15, 15))
        Qd[0:3, 0:3] = np.eye(3) * c.sigma_g**2 * dt
        Qd[3:6, 3:6] = np.eye(3) * c.sigma_a**2 * dt
        Qd[6:9, 6:9] = np.eye(3) * c.sigma_a**2 * dt**3 / 3.0
        Qd[9:12, 9:12] = np.eye(3) * c.sigma_bg**2 * dt
        Qd[12:15, 12:15] = np.eye(3) * c.sigma_ba**2 * dt

        # Clones are static under propagation, so only the core block and the
        # core/clone cross-terms change.
        P = self.P
        Pxx = P[:15, :15]
        P[:15, :15] = F @ Pxx @ F.T + Qd
        if self.n_clones:
            Pxc = F @ P[:15, 15:]
            P[:15, 15:] = Pxc
            P[15:, :15] = Pxc.T
        self.P = 0.5 * (P + P.T)

    # ---- cloning ---------------------------------------------------
    def clone(self, t):
        if self.n_clones >= self.cfg.max_clones:
            self.marginalize_oldest()
        J = np.zeros((6, self.dim))
        J[0:3, 0:3] = np.eye(3)
        J[3:6, 6:9] = np.eye(3)
        P = self.P
        top = np.hstack([P, P @ J.T])
        bot = np.hstack([J @ P, J @ P @ J.T])
        self.P = np.vstack([top, bot])
        self.P = 0.5 * (self.P + self.P.T)
        cid = self._next_cid
        self._next_cid += 1
        self.clones.append(Clone(R=self.R.copy(), p=self.p.copy(), t=t, cid=cid))
        return cid

    def marginalize_oldest(self):
        if not self.clones:
            return
        keep = np.concatenate([np.arange(15), np.arange(21, self.dim)])
        self.P = self.P[np.ix_(keep, keep)]
        self.clones.popleft()

    def _clone_slice(self, k):
        return slice(15 + 6 * k, 21 + 6 * k)

    def clone_index(self, cid):
        for k, c in enumerate(self.clones):
            if c.cid == cid:
                return k
        return None

    @property
    def oldest_cid(self):
        return self.clones[0].cid if self.clones else None

    # ---- measurement update ---------------------------------------
    def displacement_jacobian(self, k):
        cl = self.clones[k]
        R_g = cl.R_g
        dp = self.p - cl.p
        H = np.zeros((3, self.dim))
        H[:, 6:9] = R_g.T
        s = self._clone_slice(k)
        H[:, s.start + 3:s.start + 6] = -R_g.T
        # The clone frame is yaw-only, so attitude enters through d(yaw) only.
        dyaw_dtheta = yaw_world_jacobian(cl.R) @ cl.R
        H[:, s.start:s.start + 3] = -np.outer(R_g.T @ skew(E3) @ dp, dyaw_dtheta)
        return H, R_g.T @ dp

    def update_displacement(self, z, cov, k=0):
        """Fuse a learned displacement measurement expressed in clone k's frame."""
        H, h = self.displacement_jacobian(k)
        r = np.asarray(z, dtype=np.float64) - h
        Sigma = np.asarray(cov, dtype=np.float64) * self.cfg.cov_inflation
        if Sigma.ndim == 1:
            Sigma = np.diag(Sigma)
        HP = H @ self.P
        S = HP @ H.T + Sigma
        nis = float(r @ np.linalg.solve(S, r))
        if self.cfg.chi2_gate is not None and nis > self.cfg.chi2_gate:
            return {"accepted": False, "nis": nis, "residual": r}

        K = np.linalg.solve(S.T, HP).T
        dx = K @ r
        # Joseph form expanded to avoid forming (I - KH) explicitly.
        A = K @ HP
        self.P = self.P - A - A.T + K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.inject(dx)
        return {"accepted": True, "nis": nis, "residual": r}

    def update_batch(self, H, r, R):
        """Generic EKF update for a stacked measurement block."""
        HP = H @ self.P
        S = HP @ H.T + R
        K = np.linalg.solve(S.T, HP).T
        dx = K @ r
        A = K @ HP
        self.P = self.P - A - A.T + K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.inject(dx)
        return {"rows": len(r), "dx_norm": float(np.linalg.norm(dx))}

    def inject(self, dx):
        self.R = self.R @ so3_exp(dx[0:3])
        self.v = self.v + dx[3:6]
        self.p = self.p + dx[6:9]
        self.bg = self.bg + dx[9:12]
        self.ba = self.ba + dx[12:15]
        for k, cl in enumerate(self.clones):
            s = self._clone_slice(k)
            cl.R = cl.R @ so3_exp(dx[s.start:s.start + 3])
            cl.p = cl.p + dx[s.start + 3:s.start + 6]
            cl.R_g = gravity_aligned_frame(cl.R)

    # ---- diagnostics ------------------------------------------------
    def pose_covariance(self):
        idx = np.concatenate([np.arange(0, 3), np.arange(6, 9)])
        return self.P[np.ix_(idx, idx)]
