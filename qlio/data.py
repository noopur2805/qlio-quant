"""Sequence loading (TLIO golden format or synthetic) and windowed dataset."""

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import gravity_aligned_frame, quats_to_rots, skew, so3_exp

GRAVITY = np.array([0.0, 0.0, -9.81])


@dataclass
class Sequence:
    """IMU + ground truth resampled on a uniform grid, expressed in the world frame."""

    name: str
    ts: np.ndarray  # (N,) seconds
    gyr_w: np.ndarray  # (N,3) rad/s, rotated into world
    acc_w: np.ndarray  # (N,3) m/s^2 specific force, rotated into world
    R_wb: np.ndarray  # (N,3,3)
    p_w: np.ndarray  # (N,3)
    v_w: np.ndarray  # (N,3)

    @property
    def rate(self):
        return 1.0 / float(np.median(np.diff(self.ts)))

    def gyr_b(self):
        return np.einsum("nji,nj->ni", self.R_wb, self.gyr_w)

    def acc_b(self):
        return np.einsum("nji,nj->ni", self.R_wb, self.acc_w)


def _find_columns(description):
    """Map TLIO description json to column slices, keyed by semantic name."""
    names = description.get("columns_name(width)", description.get("columns_name"))
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",")]
    out, start = {}, 0
    for entry in names:
        if "(" in entry:
            key, width = entry.split("(")
            width = int(width.strip(")"))
        else:
            key, width = entry, 1
        out[key.strip()] = slice(start, start + width)
        start += width
    return out


def load_tlio_sequence(seq_dir):
    """Load one TLIO golden-format sequence directory."""
    arr = np.load(os.path.join(seq_dir, "imu0_resampled.npy")).astype(np.float64)
    with open(os.path.join(seq_dir, "imu0_resampled_description.json")) as f:
        cols = _find_columns(json.load(f))

    def col(*candidates):
        for c in candidates:
            for k, s in cols.items():
                if k.lower().startswith(c.lower()):
                    return arr[:, s]
        raise KeyError(candidates)

    ts = col("ts").reshape(-1) * 1e-6
    R_wb = quats_to_rots(col("qxyzw"))
    p_w = col("pos")
    try:
        v_w = col("vel")
    except KeyError:
        v_w = np.gradient(p_w, ts, axis=0)
    return Sequence(
        name=os.path.basename(seq_dir.rstrip("/")),
        ts=ts,
        gyr_w=col("gyr"),
        acc_w=col("acc"),
        R_wb=R_wb,
        p_w=p_w,
        v_w=v_w,
    )


def load_tlio_split(root, split):
    """Load every sequence listed in <root>/<split>_list.txt."""
    with open(os.path.join(root, f"{split}_list.txt")) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return [load_tlio_sequence(os.path.join(root, n)) for n in names]


def synth_sequence(name, duration=120.0, rate=200.0, seed=0,
                   gyr_noise=2e-3, acc_noise=2e-2,
                   gyr_bias_rw=1e-4, acc_bias_rw=1e-3):
    """Pedestrian-like trajectory with gait bobbing, turns, and a biased noisy IMU."""
    rng = np.random.default_rng(seed)
    n = int(duration * rate)
    ts = np.arange(n) / rate

    speed = 1.3 + 0.25 * np.sin(2 * np.pi * ts / 37.0 + rng.uniform(0, 6.28))
    yaw_rate = 0.35 * np.sin(2 * np.pi * ts / 23.0 + rng.uniform(0, 6.28)) \
        + 0.15 * np.sin(2 * np.pi * ts / 7.3 + rng.uniform(0, 6.28))
    yaw = np.cumsum(yaw_rate) / rate

    gait = 1.9  # Hz
    vx = speed * np.cos(yaw)
    vy = speed * np.sin(yaw)
    vz = 0.045 * 2 * np.pi * gait * np.cos(2 * np.pi * gait * ts) \
        + 0.02 * np.sin(2 * np.pi * ts / 41.0)
    v_w = np.stack([vx, vy, vz], axis=1)
    p_w = np.cumsum(v_w, axis=0) / rate

    roll = 0.06 * np.sin(2 * np.pi * gait * ts)
    pitch = 0.05 * np.sin(2 * np.pi * gait * ts + 1.1) + 0.03 * np.sin(2 * np.pi * ts / 19.0)
    R_wb = np.empty((n, 3, 3))
    for i in range(n):
        cy, sy = np.cos(yaw[i]), np.sin(yaw[i])
        cp, sp = np.cos(pitch[i]), np.sin(pitch[i])
        cr, sr = np.cos(roll[i]), np.sin(roll[i])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
        Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_wb[i] = Rz @ Ry @ Rx

    a_w = np.gradient(v_w, ts, axis=0)
    acc_w_true = a_w - GRAVITY

    dR = np.einsum("nji,njk->nik", R_wb[:-1], R_wb[1:])
    w_b = np.zeros((n, 3))
    w_b[:-1] = np.stack([
        np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) * 0.5 * rate
        for m in dR
    ])
    w_b[-1] = w_b[-2]
    gyr_w_true = np.einsum("nij,nj->ni", R_wb, w_b)

    bg = np.cumsum(rng.normal(0, gyr_bias_rw / np.sqrt(rate), (n, 3)), axis=0) + rng.normal(0, 5e-3, 3)
    ba = np.cumsum(rng.normal(0, acc_bias_rw / np.sqrt(rate), (n, 3)), axis=0) + rng.normal(0, 3e-2, 3)
    gyr_b = np.einsum("nji,nj->ni", R_wb, gyr_w_true) + bg + rng.normal(0, gyr_noise * np.sqrt(rate), (n, 3))
    acc_b = np.einsum("nji,nj->ni", R_wb, acc_w_true) + ba + rng.normal(0, acc_noise * np.sqrt(rate), (n, 3))

    return Sequence(
        name=name, ts=ts,
        gyr_w=np.einsum("nij,nj->ni", R_wb, gyr_b),
        acc_w=np.einsum("nij,nj->ni", R_wb, acc_b),
        R_wb=R_wb, p_w=p_w, v_w=v_w,
    )


def synth_split(n_seq, duration=120.0, seed=0, prefix="synth"):
    return [synth_sequence(f"{prefix}{i}", duration=duration, seed=seed + i) for i in range(n_seq)]


class WindowDataset(Dataset):
    """Fixed-length IMU windows in a gravity-aligned frame with displacement targets.

    Mirrors TLIO's setup: inputs are gyro/accel expressed in the yaw-only frame
    anchored at the window start (gravity is not removed), targets are the
    displacement over the window in that same frame.
    """

    def __init__(self, sequences, window=200, stride=20, augment=False,
                 yaw_aug=True, bias_aug=(0.01, 0.05), grav_aug_deg=5.0, seed=0):
        self.sequences = sequences
        self.window = window
        self.augment = augment
        self.yaw_aug = yaw_aug
        self.bias_aug = bias_aug
        self.grav_aug = np.deg2rad(grav_aug_deg)
        self.rng = np.random.default_rng(seed)
        self.index = [
            (si, i)
            for si, s in enumerate(sequences)
            for i in range(0, len(s.ts) - window, stride)
        ]

    def __len__(self):
        return len(self.index)

    def raw_window(self, si, i):
        s = self.sequences[si]
        j = i + self.window
        R_wg = gravity_aligned_frame(s.R_wb[i])
        gyr = s.gyr_w[i:j] @ R_wg
        acc = s.acc_w[i:j] @ R_wg
        disp = R_wg.T @ (s.p_w[j] - s.p_w[i])
        return gyr, acc, disp, R_wg

    def __getitem__(self, k):
        si, i = self.index[k]
        gyr, acc, disp, _ = self.raw_window(si, i)
        if self.augment:
            if self.bias_aug is not None:
                gyr = gyr + self.rng.normal(0, self.bias_aug[0], 3)
                acc = acc + self.rng.normal(0, self.bias_aug[1], 3)
            if self.grav_aug > 0:
                axis = self.rng.normal(size=3)
                axis[2] = 0.0
                nrm = np.linalg.norm(axis)
                if nrm > 1e-9:
                    dR = so3_exp(axis / nrm * self.rng.normal(0, self.grav_aug))
                    gyr, acc, disp = gyr @ dR, acc @ dR, dR.T @ disp
            if self.yaw_aug:
                a = self.rng.uniform(0, 2 * np.pi)
                c, s_ = np.cos(a), np.sin(a)
                Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
                gyr, acc, disp = gyr @ Rz, acc @ Rz, Rz.T @ disp
        x = np.concatenate([gyr, acc], axis=1).T.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(disp.astype(np.float32))


def calibration_batches(dataset, n=256, batch_size=32, seed=0):
    """Deterministic subsample used to fit quantization observers."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    for k in range(0, len(idx), batch_size):
        xs = [dataset[int(i)][0] for i in idx[k:k + batch_size]]
        yield torch.stack(xs)


__all__ = [
    "Sequence", "WindowDataset", "load_tlio_sequence", "load_tlio_split",
    "synth_sequence", "synth_split", "calibration_batches", "GRAVITY", "skew",
]
