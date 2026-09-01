"""Multimodal VIO: one clone window serving both camera and learned-inertial updates.

Consistency: with EKFConfig.use_fej (default) the learned-displacement update
uses first-estimates Jacobians -- the clone's measurement frame R_g and
linearisation are frozen at cloning, which measurably fixes the yaw/xy NEES
leak (see results/REPORT.md). Camera updates remain estimate-linearised:
camera-side FEJ variants were measured and did not close the leak (see
qlio.camera.feature_jacobians docstring), so their consistency caveat stands.
"""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import numpy as np

from .camera import msckf_update
from .ekf import EKFConfig, StochasticCloningEKF
from .geometry import so3_log


@dataclass
class VIOConfig:
    frame_stride: int = 10        # IMU samples between camera frames (20 Hz at 200 Hz)
    window: int = 200             # learned-inertial window length in samples
    max_clones: int = 12
    min_track_len: int = 3
    use_camera: bool = True
    use_inertial_net: bool = True
    inertial_stride: int = 10     # samples between learned updates
    overlap_inflation: bool = True
    chi2_gate_px: float = 6.0
    init_from_gt: bool = True


class TrackStore:
    """Feature observations indexed by landmark id and clone id."""

    def __init__(self):
        self.tracks = defaultdict(OrderedDict)

    def add(self, cid, obs):
        for fid, px in obs.items():
            self.tracks[fid][cid] = px

    def pop_lost(self, seen_now, min_len):
        ready, dead = [], []
        for fid, per_cid in self.tracks.items():
            if fid not in seen_now:
                dead.append(fid)
                if len(per_cid) >= min_len:
                    ready.append((fid, per_cid))
        for fid in dead:
            del self.tracks[fid]
        return ready

    def pop_using(self, cid, min_len):
        """Retire tracks whose oldest observation is about to fall off the window.

        Each track is fused at most once in its lifetime: either here, when its
        earliest clone would be marginalised, or via pop_lost, when the feature
        stops being tracked -- whichever comes first. Detaching only the
        marginalised clone and leaving the track alive (the previous behaviour)
        let the same observations be re-fused at the next marginalisation event,
        double-counting information and inflating filter NEES over time -- the
        camera-side counterpart of the overlapping-window bug on the inertial
        side.
        """
        ready, dead = [], []
        for fid, per_cid in self.tracks.items():
            if cid in per_cid:
                dead.append(fid)
                if len(per_cid) >= min_len:
                    ready.append((fid, per_cid.copy()))
        for fid in dead:
            del self.tracks[fid]
        return ready


class VIORun:
    def __init__(self):
        for k in ("t", "p_est", "p_gt", "R_est", "R_gt", "v_est", "bg", "ba",
                  "pose_err", "sigma_pose", "pose_nees", "nis", "accepted",
                  "sigma_pred", "disp_err", "n_features"):
            setattr(self, k, [])

    def finalize(self):
        for k, v in list(self.__dict__.items()):
            setattr(self, k, np.asarray(v))
        return self


def _to_tracks(ekf, ready):
    """Convert (fid, {cid: px}) records into (clone_indices, obs) for live clones."""
    out = []
    for _fid, per_cid in ready:
        ids, pts = [], []
        for cid, px in per_cid.items():
            k = ekf.clone_index(cid)
            if k is not None:
                ids.append(k)
                pts.append(px)
        if len(ids) >= 3:
            out.append((ids, np.asarray(pts)))
    return out


def run_vio(seq, frames=None, camera=None, predict=None, cfg=None, ekf_cfg=None):
    """Run the filter with camera updates, learned inertial updates, or both."""
    cfg = cfg or VIOConfig()
    ekf_cfg = ekf_cfg or EKFConfig()
    need = cfg.window // cfg.inertial_stride + 2 if cfg.use_inertial_net else 3
    ekf_cfg.max_clones = max(cfg.max_clones, need)
    if cfg.use_inertial_net and cfg.overlap_inflation:
        ekf_cfg.cov_inflation = ekf_cfg.cov_inflation * cfg.window / cfg.inertial_stride

    f = StochasticCloningEKF(ekf_cfg)
    if cfg.init_from_gt:
        f.set_state(R=seq.R_wb[0], v=seq.v_w[0], p=seq.p_w[0])

    n = len(seq.ts)
    gyr_b, acc_b = seq.gyr_b(), seq.acc_b()
    R_hist = np.empty((n, 3, 3))
    store = TrackStore()
    out = VIORun()
    clone_at = {}

    for i in range(n):
        if i > 0:
            f.propagate(gyr_b[i], acc_b[i], seq.ts[i] - seq.ts[i - 1])
        R_hist[i] = f.R

        is_clone_step = (i % cfg.inertial_stride == 0) or (i % cfg.frame_stride == 0)
        n_feat = 0
        if is_clone_step:
            if f.n_clones >= ekf_cfg.max_clones and cfg.use_camera:
                ready = store.pop_using(f.oldest_cid, cfg.min_track_len)
                tracks = _to_tracks(f, ready)
                if tracks:
                    n_feat += msckf_update(f, camera, tracks, chi2_gate=cfg.chi2_gate_px)["features"]
            cid = f.clone(seq.ts[i])
            clone_at[i] = cid

        if cfg.use_camera and frames is not None and i in frames:
            obs = frames[i]
            store.add(clone_at[i], obs)
            ready = store.pop_lost(set(obs.keys()), cfg.min_track_len)
            tracks = _to_tracks(f, ready)
            if tracks:
                n_feat += msckf_update(f, camera, tracks, chi2_gate=cfg.chi2_gate_px)["features"]

        did_inertial = False
        if (cfg.use_inertial_net and predict is not None and i >= cfg.window
                and i % cfg.inertial_stride == 0):
            i0 = i - cfg.window
            k = f.clone_index(clone_at.get(i0, -1))
            if k is not None:
                cl = f.clones[k]
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
                out.nis.append(info["nis"])
                out.accepted.append(info["accepted"])
                out.sigma_pred.append(np.sqrt(var))
                out.disp_err.append(disp - gt_disp)
                did_inertial = True

        if is_clone_step and (did_inertial or not cfg.use_inertial_net):
            dtheta = so3_log(f.R.T @ seq.R_wb[i])
            e = np.concatenate([dtheta, seq.p_w[i] - f.p])
            Ppose = f.pose_covariance()
            out.t.append(seq.ts[i])
            out.p_est.append(f.p.copy())
            out.p_gt.append(seq.p_w[i])
            out.R_est.append(f.R.copy())
            out.R_gt.append(seq.R_wb[i])
            out.v_est.append(f.v.copy())
            out.bg.append(f.bg.copy())
            out.ba.append(f.ba.copy())
            out.pose_err.append(e)
            out.sigma_pose.append(np.sqrt(np.diag(Ppose)))
            out.pose_nees.append(float(e @ np.linalg.solve(Ppose + 1e-12 * np.eye(6), e)))
            out.n_features.append(n_feat)

    return out.finalize()
