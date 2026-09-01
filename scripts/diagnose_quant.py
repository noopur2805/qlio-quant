"""Diagnostics for three suspicious patterns in the quantization results.

  1. int8-native reporting better filter consistency than fp32.
  2. mean_head-only quantization inflating mean z^2 while RMSE barely moves.
  3. Non-monotonic degradation (6 bits worse than 4 bits).

Run after a study so a trained checkpoint exists:
    PYTHONPATH=. python scripts/diagnose_quant.py --ckpt results/<run>/checkpoint_last.pt
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from qlio.data import WindowDataset, load_tlio_split, synth_split
from qlio.model import small_resnet, tlio_resnet
from qlio.quantize import (apply_fake_quant, calibrate, quantized_scopes,
                           real_static_ptq)
from qlio.train import evaluate


def log(m):
    print(m, flush=True)


def rule(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)


def load_data(args):
    if args.data_root:
        root = Path(args.data_root).expanduser()
        val = load_tlio_split(root, "val")[: args.max_seqs]
        test = load_tlio_split(root, "test")[: args.max_seqs]
    else:
        val = synth_split(1, duration=40.0, seed=200)
        test = synth_split(1, duration=60.0, seed=300)
    return (WindowDataset(val, window=200, stride=20, augment=False),
            WindowDataset(test, window=200, stride=20, augment=False))


def calib_batches(ds, n=8, bs=64):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    return [x for i, (x, _) in enumerate(dl) if i < n]


def z2_per_axis(err, sig):
    z2 = (err / np.maximum(sig, 1e-9)) ** 2
    return z2.mean(axis=0), z2.mean()


# ---------------------------------------------------------------------------
# Check 1: does the native int8 path agree with the simulated int8 path?
# ---------------------------------------------------------------------------

def check_native_vs_sim(model, val_ds, test_ds):
    rule("CHECK 1  native int8 vs simulated int8 -- do they agree?")
    batches = calib_batches(val_ds)
    x = batches[0]

    with torch.no_grad():
        m_fp, lv_fp = model(x)
    sim = calibrate(apply_fake_quant(model, w_bits=8, a_bits=8), batches)
    with torch.no_grad():
        m_sim, lv_sim = sim(x)

    nat = real_static_ptq(model, batches)
    if isinstance(nat, dict):
        log(f"  native PTQ unavailable: {nat['error']}")
        log("  -> the 'int8-native' bars cannot be reproduced on this machine.")
        return
    with torch.no_grad():
        m_nat, lv_nat = nat(x)

    for tag, lv in (("fp32", lv_fp), ("int8-sim", lv_sim), ("int8-native", lv_nat)):
        sig = torch.exp(0.5 * lv)
        log(f"  {tag:12s} sigma mean={sig.mean():.5f}  min={sig.min():.5f}  max={sig.max():.5f}")

    d_sim = (torch.exp(0.5 * lv_sim) - torch.exp(0.5 * lv_fp)).abs().mean()
    d_nat = (torch.exp(0.5 * lv_nat) - torch.exp(0.5 * lv_fp)).abs().mean()
    log(f"\n  mean |sigma - sigma_fp32|:  sim={d_sim:.6f}   native={d_nat:.6f}")
    ratio = float(torch.exp(0.5 * lv_nat).mean() / torch.exp(0.5 * lv_fp).mean())
    log(f"  native/fp32 sigma ratio: {ratio:.3f}")
    if abs(ratio - 1.0) > 0.15:
        log("  >> Native sigma is systematically scaled vs fp32. A filter fed these")
        log("     sigmas will show a shifted NEES purely from this scaling, which")
        log("     would explain int8-native 'beating' fp32.")
    if float(d_nat) > 3 * float(d_sim) or float(d_sim) > 3 * float(d_nat):
        log("  >> sim and native disagree by >3x: the simulation is not modelling")
        log("     what the native kernels actually do.")


# ---------------------------------------------------------------------------
# Check 2: do the scope masks overlap, and is z^2 driven by one axis?
# ---------------------------------------------------------------------------

def check_scope_masks(model, val_ds, test_ds):
    rule("CHECK 2  scope masks -- overlap, and per-axis z^2 breakdown")
    seen = {}
    for scope in ("stem", "blocks", "mean_head", "cov_head"):
        q = apply_fake_quant(model, scopes=(scope,), w_bits=8, a_bits=8)
        mods = quantized_scopes(q)
        log(f"  scope={scope:11s} wraps {len(mods)} layer(s): {mods}")
        for m in mods:
            seen.setdefault(m, []).append(scope)
    dupes = {m: s for m, s in seen.items() if len(s) > 1}
    log(f"\n  overlapping layers: {dupes if dupes else 'none'}")
    if not dupes:
        log("  -> masks are disjoint, so mean_head quantization cannot alter sigma.")

    batches = calib_batches(val_ds)
    err0, sig0 = evaluate(model, test_ds)
    ax0, tot0 = z2_per_axis(err0, sig0)
    rmse0 = np.sqrt(np.mean(np.sum(err0**2, axis=1)))
    log(f"\n  fp32            RMSE={rmse0:.4f}  z2/axis={np.round(ax0,3)}  mean={tot0:.3f}")

    for bits in (8, 6, 4):
        q = calibrate(apply_fake_quant(model, scopes=("mean_head",), w_bits=bits, a_bits=bits), batches)
        err, sig = evaluate(q, test_ds)
        ax, tot = z2_per_axis(err, sig)
        rmse = np.sqrt(np.mean(np.sum(err**2, axis=1)))
        dsig = np.abs(sig - sig0).mean()
        log(f"  mean_head {bits}bit RMSE={rmse:.4f}  z2/axis={np.round(ax,3)}  "
            f"mean={tot:.3f}  mean|dsigma|={dsig:.2e}")
    log("\n  If mean|dsigma| ~ 0 but z2 rises, the rise is real and comes from the")
    log("  mean error, concentrated in whichever axis has the smallest sigma --")
    log("  aggregate RMSE hides it. That is a finding, not a bug.")


# ---------------------------------------------------------------------------
# Check 3: is the non-monotonic bit-width trend just noise?
# ---------------------------------------------------------------------------

def check_monotonicity(model, val_ds, test_ds, seeds, n_calib):
    rule(f"CHECK 3  bit-width sweep over {len(seeds)} seeds (calib batches={n_calib})")
    log(f"  {'scope':12s} {'bits':>4s}  {'RMSE mean+-std':>20s}  {'z2 mean+-std':>20s}")
    for scope in ("trunk", "all"):
        scopes = ("stem", "blocks") if scope == "trunk" else ("stem", "blocks", "mean_head", "cov_head")
        for bits in (8, 6, 4):
            rm, zz = [], []
            for s in seeds:
                torch.manual_seed(s)
                g = torch.Generator().manual_seed(s)
                dl = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=True, generator=g)
                batches = [x for i, (x, _) in enumerate(dl) if i < n_calib]
                q = calibrate(apply_fake_quant(model, scopes=scopes, w_bits=bits, a_bits=bits), batches)
                err, sig = evaluate(q, test_ds)
                rm.append(np.sqrt(np.mean(np.sum(err**2, axis=1))))
                zz.append(z2_per_axis(err, sig)[1])
            log(f"  {scope:12s} {bits:4d}  {np.mean(rm):9.4f} +- {np.std(rm):<7.4f}  "
                f"{np.mean(zz):9.3f} +- {np.std(zz):<7.3f}")
    log("\n  Overlapping error bars between 6 and 4 bits mean the non-monotonicity")
    log("  is calibration noise, not a property of the bit-width.")
    log("  std == 0 exactly means calibration saw the same activation range every")
    log("  seed; lower --calib-batches to make the check sensitive.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="checkpoint_last.pt from a study run")
    p.add_argument("--model", choices=("small", "tlio"), default="tlio")
    p.add_argument("--data-root", default=None)
    p.add_argument("--max-seqs", type=int, default=2)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--calib-batches", type=int, default=2,
                   help="Keep small: if calibration covers most of the val set the "
                        "observed min/max is identical for every seed and the "
                        "seed-to-seed std collapses to zero.")
    p.add_argument("--only", choices=("1", "2", "3"), default=None)
    args = p.parse_args()

    model = tlio_resnet() if args.model == "tlio" else small_resnet()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd["model_state"] if "model_state" in sd else sd)
        log(f"loaded {args.ckpt}")
    else:
        log("WARNING: no --ckpt, using an untrained model (structure checks only)")
    model.eval()

    val_ds, test_ds = load_data(args)
    log(f"val={len(val_ds)} windows  test={len(test_ds)} windows")

    if args.only in (None, "1"):
        check_native_vs_sim(model, val_ds, test_ds)
    if args.only in (None, "2"):
        check_scope_masks(model, val_ds, test_ds)
    if args.only in (None, "3"):
        check_monotonicity(model, val_ds, test_ds, list(range(args.seeds)), args.calib_batches)


if __name__ == "__main__":
    main()
