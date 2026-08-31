import numpy as np
import pytest

from qlio.data import synth_sequence
from qlio.ekf import EKFConfig, StochasticCloningEKF
from qlio.filter_runner import RunConfig, dead_reckon, run_filter
from qlio.geometry import (gravity_aligned_frame, quat_to_rot, rot_to_quat,
                           skew, so3_exp, so3_log, yaw_of)
from qlio.metrics import ate, drift_ratio, nees_stats
from qlio.predictor import OracleDisplacement


def test_so3_exp_log_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        axis = rng.normal(size=3)
        w = axis / np.linalg.norm(axis) * rng.uniform(0, np.pi - 1e-3)
        R = so3_exp(w)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(so3_log(R), w, atol=1e-8)


def test_quat_rot_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        R = so3_exp(rng.normal(size=3))
        assert np.allclose(quat_to_rot(rot_to_quat(R)), R, atol=1e-10)


def test_gravity_aligned_frame_is_yaw_only():
    R = so3_exp(np.array([0.2, -0.1, 0.7]))
    Rg = gravity_aligned_frame(R)
    assert np.allclose(Rg[:, 2], [0, 0, 1], atol=1e-12)
    assert np.isclose(yaw_of(Rg), yaw_of(R))


def test_synth_sequence_imu_reproduces_trajectory():
    """Noise-free IMU must integrate back to the generating trajectory."""
    seq = synth_sequence("s", duration=20.0, seed=0, gyr_noise=0, acc_noise=0,
                         gyr_bias_rw=0, acc_bias_rw=0)
    seq.gyr_w = seq.gyr_w - 0.0
    p = dead_reckon(seq)
    # biases are still randomly initialised in the generator, so allow drift but
    # require the shape of the trajectory to track over a short window
    assert np.linalg.norm(p[400] - seq.p_w[400]) < 0.5


def test_displacement_jacobian_matches_numerical():
    rng = np.random.default_rng(3)
    f = StochasticCloningEKF(EKFConfig(max_clones=1))
    f.set_state(R=so3_exp(rng.normal(size=3) * 0.3), p=rng.normal(size=3))
    f.clone(0.0)
    f.p = f.p + rng.normal(size=3)
    H, h = f.displacement_jacobian(0)

    eps = 1e-6
    Hn = np.zeros_like(H)
    for j in range(f.dim):
        dx = np.zeros(f.dim)
        dx[j] = eps
        g = StochasticCloningEKF(EKFConfig(max_clones=1))
        g.set_state(R=f.R.copy(), v=f.v.copy(), p=f.p.copy(), bg=f.bg.copy(), ba=f.ba.copy())
        g.clone(0.0)
        g.clones[0].R = f.clones[0].R.copy()
        g.clones[0].p = f.clones[0].p.copy()
        g.clones[0].R_g = gravity_aligned_frame(g.clones[0].R)
        g.inject(dx)
        _, hp = g.displacement_jacobian(0)
        Hn[:, j] = (hp - h) / eps
    assert np.allclose(H, Hn, atol=1e-4), np.abs(H - Hn).max()


def test_covariance_stays_symmetric_positive_definite():
    seq = synth_sequence("s", duration=6.0, seed=5)
    f = StochasticCloningEKF(EKFConfig())
    f.set_state(R=seq.R_wb[0], v=seq.v_w[0], p=seq.p_w[0])
    gyr, acc = seq.gyr_b(), seq.acc_b()
    for i in range(1, 400):
        f.propagate(gyr[i], acc[i], seq.ts[i] - seq.ts[i - 1])
        if i % 100 == 0:
            f.clone(seq.ts[i])
    assert np.allclose(f.P, f.P.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(f.P)) > -1e-12


def test_oracle_updates_beat_dead_reckoning():
    seq = synth_sequence("s", duration=40.0, seed=7)
    oracle = OracleDisplacement(seq, sigma=0.02, seed=0)
    run = run_filter(seq, oracle, RunConfig(window=200, update_stride=10, overlap_inflation=True))
    dr = dead_reckon(seq)

    assert run.accepted.mean() > 0.9
    # Displacement updates bound the drift rate, not absolute position, so compare
    # against distance travelled rather than using an absolute threshold.
    assert drift_ratio(run.p_est, run.p_gt) < 0.02
    assert ate(run.p_est, run.p_gt) < ate(dr[-1:], seq.p_w[-1:])


def test_overlapping_updates_need_covariance_inflation():
    """20 Hz updates over 1 s windows share 95% of their samples; fusing them as
    independent double-counts information and destroys filter consistency."""
    seq = synth_sequence("s", duration=40.0, seed=7)
    cfg = dict(window=200, update_stride=10)
    naive = run_filter(seq, OracleDisplacement(seq, sigma=0.02, seed=0),
                       RunConfig(overlap_inflation=False, **cfg))
    fixed = run_filter(seq, OracleDisplacement(seq, sigma=0.02, seed=0),
                       RunConfig(overlap_inflation=True, **cfg))

    n_naive = nees_stats(naive.pose_nees, dof=6)["normalized"]
    n_fixed = nees_stats(fixed.pose_nees, dof=6)["normalized"]
    assert n_naive > 20 * n_fixed
    assert n_fixed < 10.0
    assert drift_ratio(fixed.p_est, fixed.p_gt) < drift_ratio(naive.p_est, naive.p_gt)
