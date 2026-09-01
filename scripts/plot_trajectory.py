"""3D trajectory plots from a trajectories.npz written by run_study.py / run_euroc.py.

    PYTHONPATH=. python scripts/plot_trajectory.py results/tlio_full_small
    PYTHONPATH=. python scripts/plot_trajectory.py results/euroc
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

GT = "ground_truth"
DR = "dead_reckoning"
STYLE = {GT: dict(color="k", ls="--", lw=2.0, label="ground truth"),
         DR: dict(color="#c44e52", ls=":", lw=1.2, label="dead reckoning")}
PALETTE = ["#4c72b0", "#55a868", "#8172b2", "#ccb974", "#64b5cd", "#da8bc3"]


def equal_aspect_3d(ax, pts):
    """Matplotlib has no set_aspect('equal') for 3d; force equal ranges manually
    so a drifting trajectory isn't visually distorted into looking accurate."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="directory containing trajectories.npz")
    p.add_argument("--no-dead-reckoning", action="store_true",
                   help="Drop dead reckoning; it is often orders of magnitude "
                        "larger and compresses everything else to a dot.")
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=-60.0)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    d = np.load(run_dir / "trajectories.npz")
    names = [k for k in d.files if k != "t"]
    if args.no_dead_reckoning:
        names = [n for n in names if n != DR]
    est = [n for n in names if n not in (GT, DR)]

    fig = plt.figure(figsize=(13, 5.5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    shown, ci = [], 0
    for n in names:
        pts = d[n]
        st = STYLE.get(n) or dict(color=PALETTE[ci % len(PALETTE)], lw=1.3, label=n)
        if n not in STYLE:
            ci += 1
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], **st)
        shown.append(pts)
    gt = d[GT]
    ax.scatter(*gt[0], c="green", s=45, marker="o", label="start")
    ax.scatter(*gt[-1], c="red", s=45, marker="X", label="end")
    equal_aspect_3d(ax, np.concatenate(shown))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("3D trajectory")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.legend(fontsize=8, loc="upper left")

    # Position error vs time is what actually shows divergence; the 3D view
    # alone hides it once trajectories overlap.
    ax2 = fig.add_subplot(1, 2, 2)
    t = d["t"] - d["t"][0]
    for i, n in enumerate(est):
        err = np.linalg.norm(d[n] - gt, axis=1)
        ax2.plot(t, err, lw=1.3, color=PALETTE[i % len(PALETTE)],
                 label=f"{n}  (final {err[-1]:.2f} m)")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("position error (m)")
    ax2.set_title("Absolute position error")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    fig.tight_layout()
    out = run_dir / "trajectory_3d.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
