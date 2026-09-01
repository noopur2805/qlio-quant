"""EuRoC MAV loader: real IMU + real camera images into the filter's interfaces.

Supports the ROS1 bag distribution (via the pure-python `rosbags` reader).
MH-sequence bags carry /imu0, /cam0/image_raw and /leica/position -- the
position-only ground truth. Attitude ground truth is therefore constructed as
roll/pitch from static-period gravity alignment + gyro integration (yaw=0 at
start): good enough to express body IMU in world coordinates and to initialise
the filter, but NOT an attitude reference. Only position metrics (ATE, drift)
are meaningful against the Leica track; the results JSON marks this mode.
"""

from dataclasses import dataclass

import numpy as np

from .data import Sequence
from .camera import CameraModel
from .geometry import so3_exp

# EuRoC cam0 factory calibration (identical across sequences; from sensor.yaml).
EUROC_T_BS = np.array([
    [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
    [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
    [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
    [0.0, 0.0, 0.0, 1.0],
])
EUROC_INTRINSICS = (458.654, 457.296, 367.215, 248.375)   # fu, fv, cu, cv
EUROC_DISTORTION = (-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05)
EUROC_RESOLUTION = (752, 480)


def euroc_camera(sigma_px=1.5):
    fx, fy, cx, cy = EUROC_INTRINSICS
    return CameraModel(fx=fx, fy=fy, cx=cx, cy=cy,
                       width=EUROC_RESOLUTION[0], height=EUROC_RESOLUTION[1],
                       sigma_px=sigma_px,
                       R_bc=EUROC_T_BS[:3, :3].copy(), p_bc=EUROC_T_BS[:3, 3].copy())


@dataclass
class EurocRaw:
    """Raw streams pulled out of the bag, on their native timestamps (seconds)."""
    t_imu: np.ndarray     # (N,)
    gyr: np.ndarray       # (N,3) body frame
    acc: np.ndarray       # (N,3) body frame, specific force
    t_cam: np.ndarray     # (M,) cam0 frame timestamps
    t_gt: np.ndarray      # (L,) leica timestamps
    p_gt: np.ndarray      # (L,3) leica prism position (world)


def read_bag(bag_path, load_images=False):
    """One streaming pass over the bag. Images are yielded separately by
    iter_images() to avoid holding ~3600 frames in memory."""
    from rosbags.highlevel import AnyReader
    from pathlib import Path

    t_imu, gyr, acc, t_cam, t_gt, p_gt = [], [], [], [], [], []
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections
                 if c.topic in ("/imu0", "/leica/position", "/cam0/image_raw")]
        for conn, _ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if conn.topic == "/imu0":
                t_imu.append(t)
                gyr.append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
                acc.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            elif conn.topic == "/leica/position":
                t_gt.append(t)
                p_gt.append([msg.point.x, msg.point.y, msg.point.z])
            else:
                t_cam.append(t)
    return EurocRaw(
        t_imu=np.asarray(t_imu), gyr=np.asarray(gyr), acc=np.asarray(acc),
        t_cam=np.asarray(t_cam), t_gt=np.asarray(t_gt), p_gt=np.asarray(p_gt),
    )


def iter_images(bag_path, t_start=-np.inf, t_end=np.inf):
    """Yield (t, image uint8 HxW) for cam0 frames inside [t_start, t_end]."""
    from rosbags.highlevel import AnyReader
    from pathlib import Path

    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == "/cam0/image_raw"]
        for conn, _ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if t < t_start or t > t_end:
                continue
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            yield t, img


def _static_attitude(gyr, acc, idx):
    """Roll/pitch from mean specific force over a static index range, yaw = 0."""
    g_dir = acc[idx].mean(axis=0)
    g_dir = g_dir / np.linalg.norm(g_dir)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(g_dir, z)
    s, c = np.linalg.norm(v), float(g_dir @ z)
    if s < 1e-12:
        return np.eye(3)
    return so3_exp(v / s * np.arctan2(s, c))   # R_wb: rotates body g_dir onto +z


def build_sequence(raw, t0, t1, name="euroc"):
    """Sequence on the IMU grid over [t0, t1], GT = interpolated Leica positions.

    The window must start while the platform is static: initial attitude comes
    from gravity alignment and the initial velocity is assumed zero.
    """
    m = (raw.t_imu >= t0) & (raw.t_imu <= t1)
    ts = raw.t_imu[m]
    gyr_b = raw.gyr[m]
    acc_b = raw.acc[m]

    # Static window for gravity alignment + gyro bias: the quietest 2 s slice
    # (lowest gyro power) within the first 6 s, so hand-adjustment bumps before
    # takeoff don't corrupt the initialisation.
    win = int(2.0 * 200)
    search = min(int(6.0 * 200), len(ts)) - win
    power = np.linalg.norm(gyr_b, axis=1) ** 2
    csum = np.concatenate([[0.0], np.cumsum(power)])
    start = int(np.argmin(csum[win:search + win + 1] - csum[:search + 1])) if search > 0 else 0
    idx = np.arange(start, start + win)
    gyro_bias = gyr_b[idx].mean(axis=0)
    R0 = _static_attitude(gyr_b, acc_b, idx)

    # attitude by forward integration of bias-corrected gyro
    R_wb = np.empty((len(ts), 3, 3))
    R_wb[0] = R0
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        R_wb[i] = R_wb[i - 1] @ so3_exp((gyr_b[i] - gyro_bias) * dt)

    # ground-truth position: linear interpolation of the Leica track
    p_w = np.column_stack([np.interp(ts, raw.t_gt, raw.p_gt[:, k]) for k in range(3)])
    p_w = p_w - p_w[0]  # origin at window start
    # velocity: central difference of a lightly smoothed track
    kernel = np.ones(21) / 21.0
    p_s = np.column_stack([np.convolve(p_w[:, k], kernel, mode="same") for k in range(3)])
    v_w = np.gradient(p_s, ts, axis=0)

    gyr_w = np.einsum("nij,nj->ni", R_wb, gyr_b - gyro_bias)
    acc_w = np.einsum("nij,nj->ni", R_wb, acc_b)
    return Sequence(name=name, ts=ts, gyr_w=gyr_w, acc_w=acc_w,
                    R_wb=R_wb, p_w=p_w, v_w=v_w)


def motion_start(raw, thresh=0.5):
    """First Leica timestamp of sustained motion (`thresh` metres from start;
    MH sequences show ~0.2 m of handling jitter before takeoff)."""
    d = np.linalg.norm(raw.p_gt - raw.p_gt[0], axis=1)
    i = int(np.argmax(d > thresh))
    return raw.t_gt[i] if d[i] > thresh else raw.t_gt[-1]
