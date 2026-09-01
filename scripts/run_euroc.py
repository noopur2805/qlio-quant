"""Real-data VIO: EuRoC MH_01 images through the KLT tracker into the MSCKF filter.

Camera-only VIO (no learned inertial update: the TLIO-style network is trained
on pedestrian head-mounted IMU and does not transfer to a drone) versus IMU
dead reckoning. Ground truth is the Leica position track, so position metrics
only; see qlio/euroc.py.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qlio.euroc import (EUROC_DISTORTION, build_sequence, euroc_camera,
                        iter_images, motion_start, read_bag)
from qlio.filter_runner import dead_reckon
from qlio.metrics import align_yaw_xy, ate, drift_ratio, rte
from qlio.tracker import KLTTracker
from qlio.vio_runner import VIOConfig, run_vio


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_frames(bag, seq, camera, frame_stride=10):
    """Track features over cam0 frames inside the sequence window and key them
    by IMU sample index snapped to multiples of frame_stride (run_vio clones
    only at those steps)."""
    tracker = KLTTracker(camera, EUROC_DISTORTION)
    frames = {}
    t0, t1 = seq.ts[0], seq.ts[-1]
    n_img = 0
    for t, img in iter_images(bag, t0, t1):
        obs = tracker.step(img)
        n_img += 1
        i = int(np.argmin(np.abs(seq.ts - t)))
        i = int(round(i / frame_stride)) * frame_stride
        if i >= len(seq.ts):
            continue
        if obs:
            frames[i] = obs  # last frame wins if two snap to the same index
    return frames, n_img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", default="/tmp/euroc/MH_01_easy.bag")
    ap.add_argument("--pre-motion", type=float, default=3.0,
                    help="Seconds of static data before motion onset (attitude/bias init).")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out", default="results/euroc")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log(f"reading {args.bag} (IMU + Leica + camera timestamps)")
    raw = read_bag(args.bag)
    t_move = motion_start(raw)
    t0 = t_move - args.pre_motion
    t1 = t0 + args.duration
    log(f"motion starts at t={t_move - raw.t_imu[0]:.1f}s into the bag; "
        f"window = [{t0 - raw.t_imu[0]:.1f}, {t1 - raw.t_imu[0]:.1f}]s")
    seq = build_sequence(raw, t0, t1, name="MH_01_easy")
    log(f"sequence: {len(seq.ts)} IMU samples @ {seq.rate:.0f}Hz, "
        f"GT path length {np.sum(np.linalg.norm(np.diff(seq.p_w, axis=0), axis=1)):.1f}m")

    cam = euroc_camera()
    log("tracking features (KLT + FB check + RANSAC)")
    frames, n_img = build_frames(args.bag, seq, cam)
    counts = [len(v) for v in frames.values()]
    log(f"{n_img} images -> {len(frames)} usable frames, "
        f"features/frame mean={np.mean(counts):.0f} min={np.min(counts)}")

    log("running camera-only VIO")
    run = run_vio(seq, frames=frames, camera=cam,
                  cfg=VIOConfig(use_camera=True, use_inertial_net=False))
    dr = dead_reckon(seq)

    # The filter world frame (gravity-aligned, yaw=0 at init) and the Leica
    # frame differ by an unobservable yaw: align 4-DoF before scoring, the
    # standard monocular-VIO evaluation.
    p_al = align_yaw_xy(run.p_est, run.p_gt)
    dr_al = align_yaw_xy(dr, seq.p_w)

    res = {
        "mode": "bag_leica (position-only ground truth)",
        "sequence": "MH_01_easy",
        "window_s": [float(t0 - raw.t_imu[0]), float(t1 - raw.t_imu[0])],
        "images": n_img,
        "features_per_frame": float(np.mean(counts)),
        "alignment": "4-DoF (yaw + translation)",
        "vio": {
            "ate_m": ate(p_al, run.p_gt),
            "drift_pct": drift_ratio(p_al, run.p_gt) * 100,
            "rte_10s_m": rte(run.t, p_al, run.p_gt, horizon=10.0),
            "msckf_features_fused": int(run.n_features.sum()),
        },
        "dead_reckoning": {
            "ate_m": ate(dr_al, seq.p_w),
            "final_err_m": float(np.linalg.norm(dr_al[-1] - seq.p_w[-1])),
        },
    }
    for k, v in res.items():
        log(f"  {k}: {v}")

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axs[0]
    ax.plot(run.p_gt[:, 0], run.p_gt[:, 1], "k--", lw=1.5, label="Leica ground truth")
    ax.plot(p_al[:, 0], p_al[:, 1], lw=1.2, label="camera-only VIO (aligned)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.axis("equal"); ax.legend()
    ax.set_title(f"MH_01 real images: top-down ({res['vio']['drift_pct']:.1f}% drift)")
    ax = axs[1]
    ax.plot(run.t - run.t[0], run.p_gt[:, 2], "k--", lw=1.5, label="Leica ground truth")
    ax.plot(run.t - run.t[0], p_al[:, 2], lw=1.2, label="camera-only VIO (aligned)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("z (m)"); ax.legend()
    ax.set_title("Altitude")
    fig.tight_layout()
    fig.savefig(out / "euroc_vio.png", dpi=140)

    (out / "results.json").write_text(json.dumps(res, indent=2))
    log(f"wrote {out/'results.json'} and {out/'euroc_vio.png'}")


if __name__ == "__main__":
    main()
