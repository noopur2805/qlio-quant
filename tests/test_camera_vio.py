import numpy as np
import pytest

from qlio.camera import (CameraModel, feature_jacobians, left_nullspace,
                         sample_landmarks, simulate_frames, triangulate)
from qlio.data import synth_sequence
from qlio.ekf import EKFConfig, StochasticCloningEKF
from qlio.geometry import gravity_aligned_frame, so3_exp
from qlio.metrics import drift_ratio
from qlio.vio_runner import VIOConfig, run_vio


@pytest.fixture(scope="module")
def scene():
    seq = synth_sequence("s", duration=30.0, seed=11)
    cam = CameraModel()
    lm = sample_landmarks(seq, n=500, seed=2)
    frames = simulate_frames(seq, cam, lm, frame_stride=10, seed=3)
    return seq, cam, lm, frames


def test_projection_jacobian_matches_numerical():
    cam = CameraModel()
    f_c = np.array([0.4, -0.2, 3.0])
    J = cam.proj_jacobian(f_c)
    Jn = np.zeros((2, 3))
    for k in range(3):
        d = np.zeros(3)
        d[k] = 1e-7
        Jn[:, k] = (cam.project(f_c + d) - cam.project(f_c - d)) / 2e-7
    assert np.allclose(J, Jn, rtol=1e-5, atol=1e-5)


def test_frames_have_usable_feature_tracks(scene):
    _seq, _cam, _lm, frames = scene
    counts = [len(o) for o in frames.values()]
    assert np.mean(counts) > 10, np.mean(counts)


def test_triangulation_is_exact_without_pixel_noise():
    seq = synth_sequence("s", duration=10.0, seed=11)
    cam = CameraModel(sigma_px=0.0)
    lm = sample_landmarks(seq, n=200, seed=2)
    frames = simulate_frames(seq, cam, lm, frame_stride=10, seed=3)
    idxs = sorted(frames.keys())[:12]
    seen = set.intersection(*[set(frames[i].keys()) for i in idxs[:6]])
    errs = []
    for fid in list(seen)[:20]:
        obs = [frames[i][fid] for i in idxs if fid in frames[i]]
        poses = [cam.pose_from_body(seq.R_wb[i], seq.p_w[i]) for i in idxs if fid in frames[i]]
        if len(obs) < 3:
            continue
        p_f, ok = triangulate(np.array(obs), poses, cam)
        if ok:
            errs.append(np.linalg.norm(p_f - lm[fid]))
    assert len(errs) >= 5
    assert max(errs) < 1e-6, max(errs)


def test_triangulation_precision_matches_theory(scene):
    """Depth error should sit near z^2 * sigma_px / (f * baseline)."""
    seq, cam, lm, frames = scene
    idxs = sorted(frames.keys())[:12]
    seen = set.intersection(*[set(frames[i].keys()) for i in idxs[:6]])
    assert len(seen) > 0
    errs = []
    for fid in list(seen)[:20]:
        obs, poses = [], []
        for i in idxs:
            if fid in frames[i]:
                obs.append(frames[i][fid])
                poses.append(cam.pose_from_body(seq.R_wb[i], seq.p_w[i]))
        if len(obs) < 3:
            continue
        p_f, ok = triangulate(np.array(obs), poses, cam)
        if ok:
            errs.append(np.linalg.norm(p_f - lm[fid]))
    assert len(errs) >= 5
    baseline = np.linalg.norm(seq.p_w[idxs[-1]] - seq.p_w[idxs[0]])
    depth = np.median([np.linalg.norm(lm[f] - seq.p_w[idxs[0]]) for f in list(seen)[:20]])
    expected = depth**2 * cam.sigma_px / (cam.fx * max(baseline, 1e-6))
    assert np.median(errs) < 3 * expected, (np.median(errs), expected)


def test_feature_jacobian_matches_numerical():
    rng = np.random.default_rng(4)
    cam = CameraModel()
    f = StochasticCloningEKF(EKFConfig(max_clones=3))
    f.set_state(R=so3_exp(rng.normal(size=3) * 0.2), p=rng.normal(size=3))
    for t in range(3):
        f.clone(float(t))
        f.p = f.p + rng.normal(size=3) * 0.2
    p_f = np.array([3.0, 0.5, 0.2])
    clone_ids = [0, 1, 2]
    obs = np.array([[320.0, 240.0]] * 3)
    r, Hx, _Hf = feature_jacobians(f, cam, clone_ids, obs, p_f)

    eps = 1e-6
    Hn = np.zeros_like(Hx)
    base = -r  # residual is z - h, so h = z - r
    for j in range(f.dim):
        g = StochasticCloningEKF(EKFConfig(max_clones=3))
        g.set_state(R=f.R.copy(), v=f.v.copy(), p=f.p.copy())
        for cl in f.clones:
            g.clone(cl.t)
        for gc, fc in zip(g.clones, f.clones):
            gc.R, gc.p = fc.R.copy(), fc.p.copy()
            gc.R_g = gravity_aligned_frame(gc.R)
        dx = np.zeros(g.dim)
        dx[j] = eps
        g.inject(dx)
        rp, _, _ = feature_jacobians(g, cam, clone_ids, obs, p_f)
        Hn[:, j] = ((-rp) - base) / eps
    assert np.allclose(Hx, Hn, atol=1e-3), np.abs(Hx - Hn).max()


def test_nullspace_removes_landmark_dependence():
    rng = np.random.default_rng(5)
    Hf = rng.normal(size=(8, 3))
    A = left_nullspace(Hf)
    assert A.shape == (8, 5)
    assert np.allclose(A.T @ Hf, 0.0, atol=1e-9)
    assert np.allclose(A.T @ A, np.eye(5), atol=1e-9)


def test_camera_only_vio_bounds_drift(scene):
    seq, cam, _lm, frames = scene
    run = run_vio(seq, frames=frames, camera=cam,
                  cfg=VIOConfig(use_camera=True, use_inertial_net=False))
    # Monocular VIO has no absolute position/yaw reference (4 unobservable DOF:
    # 3D position + yaw), so drift over distance travelled -- not an absolute
    # error bound -- is the right metric. 15% is a realistic bound for a
    # feature-only filter with no loop closure over this trajectory length.
    assert drift_ratio(run.p_est, run.p_gt) < 0.15, drift_ratio(run.p_est, run.p_gt)
    assert run.n_features.sum() > 50


def test_camera_only_roll_pitch_stay_consistent(scene):
    """Roll/pitch/height are observable via gravity even without loop closure,
    so -- unlike yaw/global-xy -- their NEES should stay bounded over time.
    Yaw and xy are known-unobservable and are intentionally NOT asserted here;
    see camera.py / vio_runner.py module docstrings for the FEJ/OC-EKF caveat.
    """
    seq, cam, _lm, frames = scene
    run = run_vio(seq, frames=frames, camera=cam,
                  cfg=VIOConfig(use_camera=True, use_inertial_net=False))
    n = len(run.pose_err)
    z2 = (run.pose_err / run.sigma_pose) ** 2
    observable = [0, 1, 5]  # th_x, th_y, p_z
    first_half = z2[:n // 2][:, observable].mean()
    second_half = z2[n // 2:][:, observable].mean()
    assert second_half < 10.0, second_half
    assert second_half < 3 * first_half + 2.0, (first_half, second_half)
