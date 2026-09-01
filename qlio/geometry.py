"""SO(3) utilities (numpy, float64)."""

import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(w):
    theta = float(np.linalg.norm(w))
    W = skew(w)
    if theta < 1e-8:
        return np.eye(3) + W + 0.5 * W @ W
    s, c = np.sin(theta), np.cos(theta)
    return np.eye(3) + (s / theta) * W + ((1.0 - c) / theta**2) * (W @ W)


def so3_log(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    theta = np.arccos(c)
    if theta < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def so3_right_jacobian_inv(w):
    theta = float(np.linalg.norm(w))
    W = skew(w)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * W
    a = 1.0 / theta**2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) + 0.5 * W + a * (W @ W)


def quat_to_rot(q_xyzw):
    x, y, z, w = q_xyzw
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quats_to_rots(q_xyzw):
    """Vectorised quat_to_rot: (N,4) xyzw -> (N,3,3)."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(n, 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def yaw_world_jacobian(R):
    """Row vector c such that d(yaw) = c^T phi for a world-frame perturbation exp(phi)R."""
    d = R[0, 0] ** 2 + R[1, 0] ** 2
    if d < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return np.array([-R[2, 0] * R[0, 0] / d, -R[2, 0] * R[1, 0] / d, 1.0])


def gravity_aligned_frame(R):
    """Yaw-only rotation extracted from R, i.e. the gravity-aligned body frame."""
    y = yaw_of(R)
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
