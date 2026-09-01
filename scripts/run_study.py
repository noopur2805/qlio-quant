"""End-to-end study: train the displacement+covariance net, quantize it under
different per-scope bit-widths, and measure the effect on (a) network-level
calibration and (b) downstream EKF consistency and drift. Also regenerates the
VIO/EKF diagnostic plots from the debugging session (double-counting fix,
FEJ/observability limitation, modality comparison).

Runs on CPU in a few minutes with the default sizes below. Everything is
synthetic (no external dataset needed).
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from qlio.camera import CameraModel, sample_landmarks, simulate_frames
from qlio.data import (WindowDataset, load_tlio_split, synth_sequence,
                       synth_split)
from qlio.ekf import EKFConfig
from qlio.filter_runner import RunConfig, dead_reckon, run_filter
from qlio.metrics import (ate, drift_ratio, fit_variance_scale, nees_stats,
                          overconfidence)
from qlio.model import small_resnet, tlio_resnet
from qlio.predictor import OracleDisplacement, TorchPredictor
from qlio.quantize import (apply_fake_quant, benchmark_latency, calibrate,
                           model_size_bytes)
from qlio.train import evaluate, train
from qlio.vio_runner import VIOConfig, run_vio

OUT = Path(__file__).resolve().parent.parent / "results"
PLOTS = OUT / "plots"
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_device(requested):
    if requested == "cpu":
        return "cpu"
    available = (torch.cuda.is_available() if requested.startswith("cuda")
                else getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                if requested == "mps" else False)
    if not available:
        log(f"  WARNING: requested device={requested!r} but it is not available; falling back to cpu")
        return "cpu"
    return requested


# --------------------------------------------------------------------------
# Part 1: regenerate the VIO/EKF diagnostic plots from the debugging session
# --------------------------------------------------------------------------

def part1_vio_diagnostics(seq=None):
    """Camera observations are always simulated (pinhole projection of synthetic
    landmarks with oracle association); only the underlying trajectory can be real."""
    log("Part 1: VIO/EKF diagnostics (double-counting fix, FEJ limitation, fusion)")
    if seq is None:
        seq = synth_sequence("diag", duration=30.0, seed=11)
    else:
        seq = trim_sequence(seq, 30.0)
        log(f"  trajectory from {seq.name} (real), camera observations simulated")
    cam = CameraModel()
    lm = sample_landmarks(seq, n=500, seed=2)
    frames = simulate_frames(seq, cam, lm, frame_stride=10, seed=3)
    oracle = OracleDisplacement(seq, sigma=0.02, seed=0)

    run_cam = run_vio(seq, frames=frames, camera=cam,
                      cfg=VIOConfig(use_camera=True, use_inertial_net=False))
    run_iner = run_vio(seq, predict=oracle,
                       cfg=VIOConfig(use_camera=False, use_inertial_net=True))
    run_fused = run_vio(seq, frames=frames, camera=cam, predict=oracle,
                        cfg=VIOConfig(use_camera=True, use_inertial_net=True))

    # Inertial overlap consistency: independent-20Hz vs inflated-20Hz vs 1Hz.
    ekf_indep = run_filter(seq, oracle, run_cfg=RunConfig(update_stride=10, overlap_inflation=False))
    ekf_inflated = run_filter(seq, oracle, run_cfg=RunConfig(update_stride=10, overlap_inflation=True))
    ekf_1hz = run_filter(seq, oracle, run_cfg=RunConfig(update_stride=200, overlap_inflation=False))

    fig, axs = plt.subplots(2, 3, figsize=(15.5, 9))

    ax = axs[0, 0]
    ax.plot(run_cam.p_gt[:, 0], run_cam.p_gt[:, 1], "k--", label="ground truth", lw=1.5)
    ax.plot(run_cam.p_est[:, 0], run_cam.p_est[:, 1], label="camera-only VIO", lw=1.2)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("(a) Top-down trajectory")
    ax.legend(); ax.axis("equal")

    ax = axs[0, 1]
    ax.plot(run_cam.t, run_cam.p_gt[:, 2], "k--", label="ground truth", lw=1.5)
    ax.plot(run_cam.t, run_cam.p_est[:, 2], label="camera-only VIO", lw=1.2)
    ax.set_xlabel("time (s)"); ax.set_ylabel("z / height (m)")
    ax.set_title("(a2) Altitude vs. time")
    ax.legend()

    ax = axs[0, 2]
    names = ["roll", "pitch", "yaw", "p_x", "p_y", "p_z"]
    n = len(run_cam.pose_err)
    z2 = (run_cam.pose_err / run_cam.sigma_pose) ** 2
    first = z2[:n // 2].mean(axis=0)
    second = z2[n // 2:].mean(axis=0)
    x = np.arange(6)
    ax.bar(x - 0.2, first, width=0.4, label="first half")
    ax.bar(x + 0.2, second, width=0.4, label="second half")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30)
    ax.axhline(1.0, color="green", ls=":", lw=1, label="consistent (z²=1)")
    ax.set_ylabel("mean z² = (err/σ)²")
    ax.set_title("(b) Growth concentrates in unobservable\ndirections (yaw, x, y) -- FEJ limitation")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    labels = ["camera-only", "inertial-only\n(oracle)", "fused"]
    vals = [drift_ratio(run_cam.p_est, run_cam.p_gt) * 100,
            drift_ratio(run_iner.p_est, run_iner.p_gt) * 100,
            drift_ratio(run_fused.p_est, run_fused.p_gt) * 100]
    ax.bar(labels, vals, color=["#4c72b0", "#55a868", "#c44e52"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    ax.set_ylabel("drift ratio (%)")
    ax.set_title(f"(c) Modality comparison ({seq.ts[-1]-seq.ts[0]:.0f}s {seq.name})")

    ax = axs[1, 1]
    labels = ["20Hz\nindependent\n(bug)", "20Hz\ninflated R\n(fix)", "1Hz\nnon-overlap"]
    vals = [nees_stats(ekf_indep.pose_nees, dof=6)["normalized"],
            nees_stats(ekf_inflated.pose_nees, dof=6)["normalized"],
            nees_stats(ekf_1hz.pose_nees, dof=6)["normalized"]]
    ax.bar(labels, vals, color=["#c44e52", "#55a868", "#8172b2"])
    ax.axhline(1.0, color="green", ls=":", lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    ax.set_ylabel("NEES / dof")
    ax.set_yscale("log")
    ax.set_title("(d) Inertial-side double-counting fix")

    axs[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(PLOTS / "vio_diagnostics.png", dpi=140)
    plt.close(fig)

    return {
        "camera_only_drift_pct": vals_from(run_cam),
        "inertial_only_drift_pct": drift_ratio(run_iner.p_est, run_iner.p_gt) * 100,
        "fused_drift_pct": drift_ratio(run_fused.p_est, run_fused.p_gt) * 100,
        "overlap_nees_dof": {
            "independent_bug": nees_stats(ekf_indep.pose_nees, dof=6)["normalized"],
            "inflated_fix": nees_stats(ekf_inflated.pose_nees, dof=6)["normalized"],
            "1hz_reference": nees_stats(ekf_1hz.pose_nees, dof=6)["normalized"],
        },
    }


def vals_from(run):
    return drift_ratio(run.p_est, run.p_gt) * 100


def trim_sequence(seq, duration):
    """First `duration` seconds of a sequence, so camera simulation stays tractable."""
    n = int(np.searchsorted(seq.ts - seq.ts[0], duration))
    n = max(min(n, len(seq.ts)), 2)
    return dataclasses.replace(
        seq, ts=seq.ts[:n], gyr_w=seq.gyr_w[:n], acc_w=seq.acc_w[:n],
        R_wb=seq.R_wb[:n], p_w=seq.p_w[:n], v_w=seq.v_w[:n],
    )


# --------------------------------------------------------------------------
# Part 2: train the displacement+covariance network
# --------------------------------------------------------------------------

def part2_train(args, train_seqs, val_seqs, test_seqs):
    log(f"Part 2: training {args.model}_resnet on {args.source} IMU windows "
        f"({len(train_seqs)} train / {len(val_seqs)} val / {len(test_seqs)} test sequences)")
    train_ds = WindowDataset(train_seqs, window=args.window, stride=args.stride, augment=True, seed=1)
    val_ds = WindowDataset(val_seqs, window=args.window, stride=args.stride, augment=False)
    test_ds = WindowDataset(test_seqs, window=args.window, stride=args.stride, augment=False)
    log(f"  windows: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    model = tlio_resnet() if args.model == "tlio" else small_resnet()
    device = resolve_device(args.device)
    log(f"  training on device={device}")
    t0 = time.time()
    ckpt_path = OUT / "checkpoint_last.pt" if getattr(args, "ckpt", True) else None
    history = train(model, train_ds, val_ds=val_ds, epochs_mse=args.epochs_mse,
                    epochs_nll=args.epochs_nll, batch_size=args.batch_size, seed=SEED,
                    device=device, progress=True, ckpt_path=ckpt_path)
    log(f"  trained in {(time.time()-t0)/60:.1f} min, final val_rmse_m={history[-1].get('val_rmse_m')}")
    if ckpt_path is not None:
        log(f"  checkpoint saved to {ckpt_path}")

    model = model.to("cpu")  # Parts 3-4 study single-window CPU inference specifically
    err, sig = evaluate(model, test_ds, device="cpu")
    log(f"  fp32 test RMSE={np.sqrt(np.mean(np.sum(err**2,axis=1))):.4f} m, "
        f"mean z²={overconfidence(err, sig):.2f}")
    return model, train_ds, val_ds, test_ds, history


# --------------------------------------------------------------------------
# Part 3: quantization ablation -- the actual contribution
# --------------------------------------------------------------------------

BIT_WIDTHS = (8, 6, 4)
SCOPE_SETS = {
    "mean_head-only": ("mean_head",),
    "cov_head-only": ("cov_head",),
    "trunk-only": ("stem", "blocks"),
    "all": ("stem", "blocks", "mean_head", "cov_head"),
}


def part3_quantization(model, val_ds, test_ds, test_seq):
    log("Part 3: quantization ablation across bit-widths and per-scope targets")
    from qlio.data import calibration_batches
    calib = calibration_batches(val_ds, n=128, batch_size=16, seed=0)

    x_bench = torch.randn(1, 6, 200)
    fp32_size = model_size_bytes(model)  # all scopes fp32 -> reports true fp32 size
    fp32_lat = benchmark_latency(model, x_bench)
    err0, sig0 = evaluate(model, test_ds)

    rows = []
    for scope_name, scopes in SCOPE_SETS.items():
        for bits in BIT_WIDTHS:
            qm = apply_fake_quant(model, scopes=scopes, w_bits=bits, a_bits=bits)
            calibrate(qm, calib)
            err, sig = evaluate(qm, test_ds)
            rmse = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
            z2 = overconfidence(err, sig)
            size = model_size_bytes(qm, w_bits=bits)
            lat = benchmark_latency(qm, x_bench)
            rows.append({
                "scope": scope_name, "bits": bits, "rmse_m": rmse, "mean_z2": z2,
                "size_bytes": size, "p50_ms": lat["p50_ms"], "err": err, "sig": sig,
            })
            log(f"  scope={scope_name:14s} bits={bits}  rmse={rmse:.4f}  mean_z2={z2:6.2f}  "
                f"size={size/1024:.1f}KB  p50={lat['p50_ms']:.3f}ms")

    # Recalibration: fix an int8-all-quantized model's covariance post-hoc.
    qm8 = apply_fake_quant(model, scopes=SCOPE_SETS["all"], w_bits=8, a_bits=8)
    calibrate(qm8, calib)
    err_val, sig_val = evaluate(qm8, val_ds)
    scale = fit_variance_scale(err_val, sig_val, per_axis=True)
    err_test, sig_test = evaluate(qm8, test_ds)
    sig_test_recal = sig_test * scale
    z2_recal = overconfidence(err_test, sig_test_recal)
    log(f"  int8-all + post-hoc variance recalibration: scale={scale}, "
        f"mean_z2 {overconfidence(err_test, sig_test):.2f} -> {z2_recal:.2f}")

    # --- Figure: overconfidence and RMSE vs bit-width, per scope ---
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"mean_head-only": "#c44e52", "cov_head-only": "#4c72b0",
             "trunk-only": "#55a868", "all": "#8172b2"}
    for scope_name in SCOPE_SETS:
        xs = BIT_WIDTHS
        ys = [next(r["mean_z2"] for r in rows if r["scope"] == scope_name and r["bits"] == b) for b in xs]
        axs[0].plot(xs, ys, "o-", label=scope_name, color=colors[scope_name])
    axs[0].axhline(overconfidence(err0, sig0), color="k", ls="--", lw=1, label="fp32 baseline")
    axs[0].axhline(1.0, color="green", ls=":", lw=1)
    axs[0].set_xlabel("weight/activation bits"); axs[0].set_ylabel("mean z² (1.0 = calibrated)")
    axs[0].set_title("(a) Quantizing the covariance head\nover-confidence grows sharply")
    axs[0].invert_xaxis(); axs[0].legend(fontsize=8)

    for scope_name in SCOPE_SETS:
        xs = BIT_WIDTHS
        ys = [next(r["rmse_m"] for r in rows if r["scope"] == scope_name and r["bits"] == b) for b in xs]
        axs[1].plot(xs, ys, "o-", label=scope_name, color=colors[scope_name])
    axs[1].axhline(float(np.sqrt(np.mean(np.sum(err0**2, axis=1)))), color="k", ls="--", lw=1, label="fp32 baseline")
    axs[1].set_xlabel("weight/activation bits"); axs[1].set_ylabel("displacement RMSE (m)")
    axs[1].set_title("(b) ...while displacement RMSE\nbarely moves -- the danger is invisible")
    axs[1].invert_xaxis(); axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "quant_uncertainty.png", dpi=140)
    plt.close(fig)

    # --- Figure: calibration curve fp32 vs int8 vs int8+recalibrated ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, e, s, style in [
        ("fp32", err0, sig0, "-o"),
        ("int8 (all scopes)", err_test, sig_test, "-s"),
        ("int8 + variance recal.", err_test, sig_test_recal, "-^"),
    ]:
        order = np.argsort(s.mean(axis=1))
        bins = np.array_split(order, 8)
        px, py = [], []
        for b in bins:
            if len(b) == 0:
                continue
            px.append(float(np.mean(s[b])))
            py.append(float(np.sqrt(np.mean(np.sum(e[b] ** 2, axis=1) / e.shape[1]))))
        ax.plot(px, py, style, label=label)
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", lw=1, label="perfectly calibrated")
    ax.set_xlabel("predicted σ (m)"); ax.set_ylabel("realized RMS error (m)")
    ax.set_title("Covariance calibration: predicted vs. realized error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "quant_calibration.png", dpi=140)
    plt.close(fig)

    # --- Figure: deployment cost (size, latency) for the "all" scope sweep ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    xs = ["fp32"] + [f"int{b}" for b in BIT_WIDTHS]
    sizes = [fp32_size] + [next(r["size_bytes"] for r in rows if r["scope"] == "all" and r["bits"] == b) for b in BIT_WIDTHS]
    lats = [fp32_lat["p50_ms"]] + [next(r["p50_ms"] for r in rows if r["scope"] == "all" and r["bits"] == b) for b in BIT_WIDTHS]
    axs[0].bar(xs, np.array(sizes) / 1024, color="#4c72b0")
    axs[0].set_ylabel("model size (KB)"); axs[0].set_title("(a) Parameter storage")
    axs[1].bar(xs, lats, color="#55a868")
    axs[1].set_ylabel("p50 latency (ms), batch=1, CPU"); axs[1].set_title("(b) Inference latency")
    fig.tight_layout()
    fig.savefig(PLOTS / "quant_deploy.png", dpi=140)
    plt.close(fig)

    return {
        "fp32": {"rmse_m": float(np.sqrt(np.mean(np.sum(err0**2, axis=1)))),
                "mean_z2": overconfidence(err0, sig0),
                "size_bytes": fp32_size, "p50_ms": fp32_lat["p50_ms"]},
        "sweep": [{k: v for k, v in r.items() if k not in ("err", "sig")} for r in rows],
        "recalibration": {"scale": scale.tolist(),
                          "mean_z2_before": overconfidence(err_test, sig_test),
                          "mean_z2_after": z2_recal},
    }


# --------------------------------------------------------------------------
# Part 4: does the quantized network still work inside the EKF?
# --------------------------------------------------------------------------

def part4_ekf_with_quantized_net(model, val_ds, test_seq):
    log("Part 4: quantized network as an EKF measurement source")
    from qlio.data import calibration_batches
    calib = calibration_batches(val_ds, n=128, batch_size=16, seed=0)

    configs = {}
    configs["fp32"] = TorchPredictor(model)

    qm8 = apply_fake_quant(model, scopes=SCOPE_SETS["all"], w_bits=8, a_bits=8)
    calibrate(qm8, calib)
    configs["int8-all"] = TorchPredictor(qm8)

    err_val, sig_val = evaluate(qm8, val_ds)
    scale = fit_variance_scale(err_val, sig_val, per_axis=True)
    configs["int8-all+recal"] = TorchPredictor(qm8, sigma_scale=scale)

    qmix = apply_fake_quant(model, scopes=SCOPE_SETS["trunk-only"], w_bits=8, a_bits=8)
    calibrate(qmix, calib)
    configs["int8-trunk (cov fp32)"] = TorchPredictor(qmix)

    dr = dead_reckon(test_seq)
    results = {"dead_reckoning_ate_m": ate(dr, test_seq.p_w)}
    for name, predictor in configs.items():
        run = run_filter(test_seq, predictor, run_cfg=RunConfig(update_stride=10, overlap_inflation=True))
        results[name] = {
            "ate_m": ate(run.p_est, run.p_gt),
            "drift_pct": drift_ratio(run.p_est, run.p_gt) * 100,
            "nees_dof": nees_stats(run.pose_nees, dof=6)["normalized"],
        }
        log(f"  {name:24s} ATE={results[name]['ate_m']:.3f}m  drift={results[name]['drift_pct']:.2f}%  "
            f"NEES/dof={results[name]['nees_dof']:.2f}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(configs.keys())
    drifts = [results[n]["drift_pct"] for n in names]
    nees = [results[n]["nees_dof"] for n in names]
    colors = ["#4c72b0", "#c44e52", "#55a868", "#8172b2"]
    axs[0].bar(names, drifts, color=colors)
    axs[0].axhline(results["dead_reckoning_ate_m"] and drift_ratio(dr, test_seq.p_w) * 100,
                  color="k", ls="--", lw=1, label="dead reckoning")
    axs[0].set_ylabel("drift ratio (%)"); axs[0].set_title("(a) EKF drift by precision config")
    axs[0].tick_params(axis="x", rotation=20); axs[0].legend(fontsize=8)

    axs[1].bar(names, nees, color=colors)
    axs[1].axhline(1.0, color="green", ls=":", lw=1)
    axs[1].set_ylabel("NEES / dof"); axs[1].set_title("(b) Filter consistency by precision config")
    axs[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOTS / "quant_ekf_consistency.png", dpi=140)
    plt.close(fig)

    return results


def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None,
                   help="TLIO golden-format root (contains train_list.txt etc). "
                        "Omit to use the synthetic generator.")
    p.add_argument("--max-train-seqs", type=int, default=8)
    p.add_argument("--max-val-seqs", type=int, default=2)
    p.add_argument("--max-test-seqs", type=int, default=2)
    p.add_argument("--model", choices=("small", "tlio"), default="small")
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--epochs-mse", type=int, default=2)
    p.add_argument("--epochs-nll", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu",
                   help="Device for training/evaluation in Part 2, e.g. cuda, cuda:0, mps. "
                        "Parts 3-4 (quantization, EKF) always run on CPU: that IS the "
                        "deployment scenario being studied (single-window inference).")
    p.add_argument("--skip-camera", action="store_true",
                   help="Skip part 1; camera observations are simulated even on real IMU data.")
    p.add_argument("--no-ckpt", dest="ckpt", action="store_false",
                   help="Disable per-epoch checkpoint saving (default: enabled).")
    p.add_argument("--out", default=None, help="Output directory (default: <repo>/results)")
    return p.parse_args()


def load_sequences(args):
    if args.data_root:
        root = Path(args.data_root).expanduser()
        log(f"Loading TLIO golden data from {root}")
        tr = load_tlio_split(root, "train")[: args.max_train_seqs]
        va = load_tlio_split(root, "val")[: args.max_val_seqs]
        te = load_tlio_split(root, "test")[: args.max_test_seqs]
        for s in tr[:1] + va[:1] + te[:1]:
            log(f"  {s.name}: {len(s.ts)} samples, {s.ts[-1]-s.ts[0]:.1f}s, {s.rate:.1f} Hz")
        return tr, va, te
    log("Using the synthetic pedestrian generator (no real data)")
    return (synth_split(4, duration=40.0, seed=100),
            synth_split(1, duration=40.0, seed=200),
            synth_split(1, duration=60.0, seed=300))


def main():
    global OUT, PLOTS
    args = build_args()
    args.source = "real (TLIO)" if args.data_root else "synthetic"
    if args.out:
        OUT = Path(args.out)
        PLOTS = OUT / "plots"
    PLOTS.mkdir(parents=True, exist_ok=True)

    train_seqs, val_seqs, test_seqs = load_sequences(args)
    test_seq = test_seqs[0]

    diag = None if args.skip_camera else part1_vio_diagnostics(
        seq=test_seq if args.data_root else None)
    model, train_ds, val_ds, test_ds, history = part2_train(args, train_seqs, val_seqs, test_seqs)
    quant = part3_quantization(model, val_ds, test_ds, test_seq)
    ekf_quant = part4_ekf_with_quantized_net(model, val_ds, test_seq)

    report = {"config": vars(args), "vio_diagnostics": diag, "training_history": history,
              "quantization": quant, "ekf_with_quantized_net": ekf_quant}
    (OUT / "results.json").write_text(json.dumps(report, indent=2, default=float))
    log(f"Wrote {OUT/'results.json'} and plots to {PLOTS}")


if __name__ == "__main__":
    main()
