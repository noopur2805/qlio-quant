"""Pinhole camera model, landmark simulation, triangulation, and MSCKF updates."""

from dataclasses import dataclass, field

import numpy as np

from .geometry import skew


@dataclass
class CameraModel:
    fx: float = 240.0
    fy: float = 240.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    sigma_px: float = 1.0
    R_bc: np.ndarray = field(default_factory=lambda: np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))  # body x-forward -> camera z-forward
    p_bc: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.0, 0.02]))

    def pose_from_body(self, R_wb, p_wb):
        return R_wb @ self.R_bc, p_wb + R_wb @ self.p_bc

    def project(self, f_c):
        z = f_c[2]
        return np.array([self.fx * f_c[0] / z + self.cx, self.fy * f_c[1] / z + self.cy])

    def proj_jacobian(self, f_c):
        x, y, z = f_c
        return np.array([
            [self.fx / z, 0.0, -self.fx * x / z**2],
            [0.0, self.fy / z, -self.fy * y / z**2],
        ])

    def in_view(self, f_c, min_depth=0.3, max_depth=40.0):
        if not (min_depth < f_c[2] < max_depth):
            return False
        u, v = self.project(f_c)
        return 0 <= u < self.width and 0 <= v < self.height


def sample_landmarks(seq, n=600, radius=8.0, height=(-1.5, 2.5), seed=0):
    """Scatter 3D points in a tube around the trajectory."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seq.p_w), n)
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = radius * np.sqrt(rng.uniform(0.15, 1.0, n))
    pts = seq.p_w[idx].copy()
    pts[:, 0] += rad * np.cos(ang)
    pts[:, 1] += rad * np.sin(ang)
    pts[:, 2] = rng.uniform(*height, n)
    return pts


def simulate_frames(seq, camera, landmarks, frame_stride=10, seed=0, outlier_rate=0.0):
    """Noisy pixel observations of visible landmarks at every frame_stride sample.

    Returns {sample_index: {landmark_id: pixel}}.
    """
    rng = np.random.default_rng(seed)
    frames = {}
    for i in range(0, len(seq.ts), frame_stride):
        R_wc, p_wc = camera.pose_from_body(seq.R_wb[i], seq.p_w[i])
        f_c = (landmarks - p_wc) @ R_wc
        obs = {}
        for j, fc in enumerate(f_c):
            if camera.in_view(fc):
                px = camera.project(fc) + rng.normal(0, camera.sigma_px, 2)
                if outlier_rate and rng.random() < outlier_rate:
                    px = px + rng.normal(0, 25.0, 2)
                obs[j] = px
        frames[i] = obs
    return frames


def triangulate(obs, poses, camera, iters=8):
    """Inverse-depth Gauss-Newton triangulation in the first observing camera frame.

    obs: (M,2) pixels; poses: list of (R_wc, p_wc). Returns (p_w, ok).

    Vectorised across the M observations. R_k0/t_k0 depend only on the (fixed)
    camera poses, not on the unknown inverse-depth parameters, so they are
    computed once before the Gauss-Newton loop rather than recomputed on every
    iteration -- the latter was previously the dominant cost of triangulation.
    """
    obs = np.asarray(obs, dtype=float)
    M = len(obs)
    if M < 2:
        return None, False
    R0, p0 = poses[0]
    bearings = np.column_stack([(obs[:, 0] - camera.cx) / camera.fx,
                                (obs[:, 1] - camera.cy) / camera.fy,
                                np.ones(M)])
    bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)

    p_arr = np.array([p for _, p in poses])
    base = np.linalg.norm(p_arr - p0, axis=1)
    if base[1:].max(initial=0.0) < 1e-3:
        return None, False
    best = int(np.argmax(base))

    r_k0 = np.stack([Rk.T @ R0 for Rk, _ in poses])          # (M,3,3), fixed across iters
    t_k0 = np.stack([Rk.T @ (p0 - pk) for Rk, pk in poses])  # (M,3)

    r_0k = R0.T @ poses[best][0]
    t_0k = R0.T @ (poses[best][1] - p0)
    b0, bk = bearings[0], r_0k @ bearings[best]
    a_mat = np.stack([b0, -bk], axis=1)
    try:
        lam, _res, rank, _ = np.linalg.lstsq(a_mat, t_0k, rcond=None)
    except np.linalg.LinAlgError:
        return None, False
    if rank < 2 or lam[0] <= 1e-3:
        return None, False
    x = b0 * lam[0]
    theta = np.array([x[0] / x[2], x[1] / x[2], 1.0 / x[2]])

    dh = np.stack([r_k0[:, :, 0], r_k0[:, :, 1], t_k0], axis=2)  # (M,3,3): d h / d theta

    for _ in range(iters):
        a, b, rho = theta
        h = r_k0 @ np.array([a, b, 1.0]) + rho * t_k0  # (M,3)
        if np.any(h[:, 2] < 1e-6):
            return None, False
        pred = np.column_stack([camera.fx * h[:, 0] / h[:, 2] + camera.cx,
                                camera.fy * h[:, 1] / h[:, 2] + camera.cy])
        r = obs - pred  # (M,2)

        jp = np.zeros((M, 2, 3))
        jp[:, 0, 0] = camera.fx / h[:, 2]
        jp[:, 0, 2] = -camera.fx * h[:, 0] / h[:, 2] ** 2
        jp[:, 1, 1] = camera.fy / h[:, 2]
        jp[:, 1, 2] = -camera.fy * h[:, 1] / h[:, 2] ** 2
        j = np.einsum("mij,mjk->mik", jp, dh)  # (M,2,3)

        jtj = np.einsum("mij,mik->jk", j, j)
        jtr = np.einsum("mij,mi->j", j, r)
        try:
            theta = theta + np.linalg.solve(jtj + 1e-9 * np.eye(3), jtr)
        except np.linalg.LinAlgError:
            return None, False
        if theta[2] <= 1e-6:
            return None, False

    a, b, rho = theta
    p_w = p0 + R0 @ (np.array([a, b, 1.0]) / rho)
    depth = 1.0 / rho
    return p_w, bool(0.3 < depth < 100.0)


def feature_jacobians(ekf, camera, clone_ids, obs, p_f):
    """Stack per-observation residuals and Jacobians w.r.t. error state and landmark.

    Linearised at the current clone estimates. First-estimate (FEJ) variants of
    this update were implemented and measured (H at frozen clone poses, with
    and without fej-consistent triangulation): accuracy was a wash but the
    yaw/xy consistency leak did not close, because the residual's landmark
    sensitivity is evaluated at the estimate while the nullspace projection
    would be built at the first estimates. Closing it properly needs
    observability-constrained updates (OC-EKF), which is out of scope; the
    learned-displacement update uses FEJ (see qlio.ekf) where it demonstrably
    fixes consistency.
    """
    M = len(clone_ids)
    r = np.zeros(2 * M)
    Hx = np.zeros((2 * M, ekf.dim))
    Hf = np.zeros((2 * M, 3))
    for m, (k, px) in enumerate(zip(clone_ids, obs)):
        cl = ekf.clones[k]
        R_wc, p_wc = camera.pose_from_body(cl.R, cl.p)
        f_c = R_wc.T @ (p_f - p_wc)
        if f_c[2] < 1e-6:
            return None, None, None
        Jp = camera.proj_jacobian(f_c)
        r[2 * m:2 * m + 2] = px - camera.project(f_c)
        s = ekf._clone_slice(k)
        Hx[2 * m:2 * m + 2, s.start:s.start + 3] = Jp @ camera.R_bc.T @ (skew(cl.R.T @ (p_f - p_wc)) + skew(camera.p_bc))
        Hx[2 * m:2 * m + 2, s.start + 3:s.start + 6] = -Jp @ R_wc.T
        Hf[2 * m:2 * m + 2] = Jp @ R_wc.T
    return r, Hx, Hf


def left_nullspace(Hf):
    u, s, _ = np.linalg.svd(Hf, full_matrices=True)
    rank = int(np.sum(s > 1e-9))
    return u[:, rank:]


def msckf_update(ekf, camera, tracks, chi2_gate=None, max_rows=200, max_features=40):
    """Fuse completed feature tracks with landmark states marginalised out.

    tracks: list of (clone_ids, obs array (M,2)).
    max_features caps the number of tracks triangulated per call (longest
    tracks first, since they carry the most information per unit of compute).
    max_rows caps the stacked measurement size via QR compression (MSCKF III-D)
    so a single update never costs more than a fixed, bounded solve.
    """
    if max_features is not None and len(tracks) > max_features:
        tracks = sorted(tracks, key=lambda t: -len(t[0]))[:max_features]

    rows_H, rows_r = [], []
    used = 0
    for clone_ids, obs in tracks:
        if len(clone_ids) < 3:
            continue
        poses = [camera.pose_from_body(ekf.clones[k].R, ekf.clones[k].p) for k in clone_ids]
        p_f, ok = triangulate(obs, poses, camera)
        if not ok:
            continue
        r, Hx, Hf = feature_jacobians(ekf, camera, clone_ids, obs, p_f)
        if r is None:
            continue
        A = left_nullspace(Hf)
        if A.shape[1] == 0:
            continue
        r_o = A.T @ r
        H_o = A.T @ Hx
        R_o = np.eye(len(r_o)) * camera.sigma_px**2
        S = H_o @ ekf.P @ H_o.T + R_o
        gamma = float(r_o @ np.linalg.solve(S, r_o))
        if chi2_gate is not None and gamma > chi2_gate * len(r_o):
            continue
        rows_H.append(H_o)
        rows_r.append(r_o)
        used += 1

    if not rows_H:
        return {"features": 0, "rows": 0}

    H = np.vstack(rows_H)
    r = np.concatenate(rows_r)
    if max_rows is not None and H.shape[0] > max_rows:
        H, r = qr_compress(H, r)
    R = np.eye(len(r)) * camera.sigma_px**2
    ekf.update_batch(H, r, R)
    return {"features": used, "rows": int(H.shape[0])}


def qr_compress(H, r):
    """Thin-QR reduction of a tall measurement stack (MSCKF section III-D)."""
    Q, Rm = np.linalg.qr(H, mode="reduced")
    return Rm, Q.T @ r
