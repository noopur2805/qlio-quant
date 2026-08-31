# qlio-quant — Quantized Learned Inertial Odometry

A stochastic-cloning EKF (MSCKF-style) visual-inertial odometry system, with a
TLIO-style learned displacement+covariance network fused in as an additional
inertial measurement update, plus a quantization ablation studying how
int-N compression of that network affects filter consistency (NEES) and drift.

- `qlio/geometry.py` — SO(3) utilities, exact yaw-world Jacobians
- `qlio/data.py` — TLIO golden-format loader + synthetic pedestrian IMU/trajectory generator
- `qlio/model.py` — 1D ResNet with separate mean/log-variance heads
- `qlio/ekf.py` — stochastic-cloning EKF, Joseph-form updates, chi-squared gating
- `qlio/camera.py` — MSCKF camera updates: inverse-depth triangulation, nullspace projection
- `qlio/vio_runner.py` — fuses camera + learned-inertial updates in one filter
- `qlio/quantize.py` — fake-quantization, calibration, size/latency benchmarking
- `qlio/train.py`, `qlio/losses.py` — two-stage (MSE → Gaussian NLL) training
- `qlio/metrics.py` — ATE, drift ratio, NEES, overconfidence (mean z²), variance recalibration
- `scripts/run_study.py` — end-to-end study: diagnostics → train → quantize → re-run EKF
- `tests/` — 16 unit tests covering geometry, EKF, and camera/MSCKF paths

**Important caveat:** the camera path (`qlio/camera.py`) always uses *simulated*
observations — pinhole projection of synthetic landmarks with oracle data
association, no real images, no feature tracker. This is true even when the
underlying trajectory/IMU comes from real TLIO data. Use `--skip-camera` when
running on real data to avoid mixing a real trajectory with fake vision.

---

## Setup

### Option A: venv (no extra installs needed)

```bash
git clone https://github.com/noopur2805/qlio-quant.git
cd qlio-quant
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: conda

```bash
git clone https://github.com/noopur2805/qlio-quant.git
cd qlio-quant
conda create -n qlio python=3.11 -y
conda activate qlio
pip install -r requirements.txt
```

### GPU (NVIDIA/CUDA)

The default `requirements.txt` install pulls CPU-only PyTorch. If you have an
NVIDIA GPU, replace the torch wheel after installing requirements:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If you hit `undefined symbol: iJIT_NotifyEvent` on conda, it's a conda MKL/ITT
ABI conflict, not your code — fix by force-reinstalling torch via pip inside
the same env:

```bash
conda remove -y mkl intel-openmp mkl-service --force
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Sanity check (synthetic data, no download, ~1 min)

```bash
PYTHONPATH=. python -m pytest tests -q
PYTHONPATH=. python scripts/run_study.py --out results/synthetic
```

Outputs: `results/synthetic/results.json`, `results/synthetic/plots/*.png`.

---

## Real data: TLIO golden dataset

IMU-only pedestrian dataset released with the TLIO paper. No images — camera
path stays simulated regardless.

```bash
mkdir -p local_data
gdown 14YKW7PsozjHo_EdxivKvumsQB7JMw1eg
unzip golden-new-format-cc-by-nc-with-imus-v1.5.zip -d local_data/
rm golden-new-format-cc-by-nc-with-imus-v1.5.zip
```

Check disk space first — this is a large download.

Expected layout (`--data-root` points here):

```
local_data/tlio_golden/
├── <sequence_id>/
│   ├── imu0_resampled.npy
│   ├── imu0_resampled_description.json
│   └── calibration.json
├── train_list.txt
├── val_list.txt
└── test_list.txt
```

### Smoke test (few sequences, CPU, ~2 min)

```bash
PYTHONPATH=. python scripts/run_study.py \
  --data-root local_data/tlio_golden \
  --max-train-seqs 8 --max-val-seqs 2 --max-test-seqs 2 \
  --skip-camera \
  --out results/tlio_smoke
```

### Full run, CPU (smaller network)

```bash
PYTHONPATH=. python scripts/run_study.py \
  --data-root local_data/tlio_golden \
  --model small \
  --max-train-seqs 200 --max-val-seqs 20 --max-test-seqs 20 \
  --epochs-mse 2 --epochs-nll 5 --batch-size 128 \
  --skip-camera \
  --out results/tlio_cpu_full
```

### Full run, GPU (paper-sized network)

```bash
PYTHONPATH=. python scripts/run_study.py \
  --data-root local_data/tlio_golden \
  --model tlio \
  --max-train-seqs 200 --max-val-seqs 20 --max-test-seqs 20 \
  --epochs-mse 5 --epochs-nll 25 --batch-size 128 \
  --skip-camera \
  --out results/tlio_full
```

Drop `--batch-size` to 64 or 32 if you hit an out-of-memory error (e.g. on a
6 GB laptop GPU).

---

## `run_study.py` options

```
--data-root PATH       TLIO golden-format root. Omit to use the synthetic generator.
--max-train-seqs N     cap on training sequences (default 8)
--max-val-seqs N       cap on validation sequences (default 2)
--max-test-seqs N      cap on test sequences (default 2)
--model {small,tlio}   network size (default small)
--window N             IMU window length in samples (default 200 = 1s @ 200Hz)
--stride N             window stride in samples (default 20)
--epochs-mse N         MSE warm-up epochs (default 2)
--epochs-nll N         Gaussian NLL epochs (default 5)
--batch-size N         (default 64)
--device DEV           device for training/eval in Part 2 only, e.g. cuda, cuda:0, mps
                       (default cpu). Parts 3-4 (quantization, EKF) always run on CPU:
                       that IS the deployment scenario being studied.
--skip-camera          skip Part 1 camera/MSCKF diagnostics (recommended on real data)
--out DIR              output directory (default: results/)
```

Example GPU training run: add `--device cuda` to any of the commands above.

---

## What the study produces

`results/<name>/results.json` and five plots in `results/<name>/plots/`:

1. `vio_diagnostics.png` — camera-only trajectory, per-axis error growth,
   modality comparison (camera-only vs inertial-only vs fused drift)
2. `quant_uncertainty.png` — RMSE and calibration (mean z²) vs. bit-width,
   per quantization scope (mean head / cov head / trunk / all)
3. `quant_calibration.png` — before/after variance recalibration
4. `quant_deploy.png` — model size and latency vs. bit-width
5. `quant_ekf_consistency.png` — ATE, drift %, NEES/dof for fp32 vs.
   quantized nets used as EKF measurement sources
