# Quantized Learned Inertial Odometry

A stochastic-cloning EKF (MSCKF-style) visual-inertial odometry system, with a
TLIO-style learned displacement+covariance network fused in as an additional
inertial measurement update, plus a quantization ablation studying how
int-N compression of that network affects filter consistency (NEES) and drift.

Headline measured results (details + caveats in `results/REPORT.md`):

- **FEJ consistency fix** on the learned-displacement update: NEES 3.9 → 0.35,
  drift 0.60 % → 0.13 % on the diagnostic A/B (`EKFConfig.use_fej`, default on).
- **Full TLIO dataset** (283 train / 36 test sequences): fp32 RMSE 0.231 m at
  z² 1.19; int4 on the mean head silently destroys calibration (z² 5) while
  RMSE barely moves; native int8 PTQ runs **2.2× faster** on real kernels.
- **Real images**: KLT front-end on EuRoC MH_01 through the same filter —
  **0.93 % drift / 0.33 m ATE over a 135 s flight** (dead reckoning: 7.5 km).
- **C++ core** (Eigen + pybind11) with lockstep parity to 1e-10 and 3.3×
  throughput.

- `qlio/geometry.py` — SO(3) utilities, exact yaw-world Jacobians
- `qlio/data.py` — TLIO golden-format loader + synthetic pedestrian IMU/trajectory generator
- `qlio/model.py` — 1D ResNet with separate mean/log-variance heads
- `qlio/ekf.py` — stochastic-cloning EKF, FEJ, Joseph-form updates, chi-squared gating
- `qlio/camera.py` — MSCKF camera updates: inverse-depth triangulation, nullspace projection
- `qlio/vio_runner.py` — fuses camera + learned-inertial updates in one filter
- `qlio/tracker.py` — real KLT feature tracker (OpenCV): FB check, RANSAC, undistortion
- `qlio/euroc.py` — EuRoC MAV rosbag loader (images streamed, Leica ground truth)
- `qlio/quantize.py` — fake-quant per scope + native int8 static PTQ, benchmarking
- `qlio/train.py`, `qlio/losses.py` — two-stage (MSE → Gaussian NLL) training
- `qlio/metrics.py` — ATE, drift ratio, NEES, mean z², 4-DoF alignment, recalibration
- `scripts/run_study.py` — end-to-end study: diagnostics → train → quantize → re-run EKF
- `scripts/run_euroc.py` — real-image VIO on EuRoC MH_01
- `cpp/` — C++ EKF core (Eigen + pybind11) with parity check and benchmark
- `tests/` — 18 unit tests covering geometry, EKF/FEJ, and camera/MSCKF paths

**Caveat on the TLIO study:** the TLIO dataset is IMU-only, so `run_study.py`'s
camera path uses *simulated* observations (pinhole projection of synthetic
landmarks, oracle association). Use `--skip-camera` on real TLIO data to avoid
mixing a real trajectory with fake vision. For vision on **real images**, use
`scripts/run_euroc.py` — real camera stream, real KLT tracker, no oracle.

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

### Full dataset, CPU (small network — the run behind results/REPORT.md §2-3)

```bash
PYTHONPATH=. python scripts/run_study.py \
  --data-root local_data/tlio_golden \
  --model small \
  --max-train-seqs 400 --max-val-seqs 40 --max-test-seqs 40 \
  --epochs-mse 2 --epochs-nll 6 --batch-size 128 \
  --skip-camera \
  --out results/tlio_full_small
```

### Full dataset, GPU (paper-sized network)

```bash
PYTHONPATH=. python scripts/run_study.py \
  --data-root local_data/tlio_golden \
  --model tlio --device cuda \
  --max-train-seqs 400 --max-val-seqs 40 --max-test-seqs 40 \
  --epochs-mse 5 --epochs-nll 25 --batch-size 128 \
  --skip-camera \
  --out results/tlio_full
```

Drop `--batch-size` to 64 or 32 if you hit an out-of-memory error (e.g. on a
6 GB laptop GPU).

---

## Real images: EuRoC MH_01 (KLT tracker + MSCKF)

```bash
pip install opencv-python-headless rosbags
# official zip server is often down; the HuggingFace mirror hosts the rosbag:
curl -L -o /tmp/euroc/MH_01_easy.bag --create-dirs \
  "https://huggingface.co/datasets/kavehsgh/EuRoC_MAV_Dataset_Machine_Hall_Easy_01/resolve/main/MH_01_easy.bag"
PYTHONPATH=. python scripts/run_euroc.py --bag /tmp/euroc/MH_01_easy.bag \
  --duration 135 --out results/euroc
```

Outputs `results/euroc/euroc_vio.png` (top-down + altitude vs Leica ground
truth) and `results/euroc/results.json`. Ground truth is position-only, so
metrics are computed after standard 4-DoF (yaw+translation) alignment.

---

## C++ EKF core

```bash
sudo apt-get install libeigen3-dev
pip install pybind11
bash cpp/build.sh
PYTHONPATH=. python cpp/parity_check.py   # lockstep parity vs python, atol=1e-10
PYTHONPATH=. python cpp/bench.py          # ~3.3x on the mixed filter schedule
```

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
   modality comparison, double-counting fix, FEJ vs legacy consistency A/B
2. `quant_uncertainty.png` — RMSE and calibration (mean z²) vs. bit-width,
   per quantization scope (mean head / cov head / trunk / all)
3. `quant_calibration.png` — before/after variance recalibration
4. `quant_deploy.png` — model size and latency vs. bit-width
5. `quant_ekf_consistency.png` — ATE, drift %, NEES/dof for fp32 vs.
   quantized nets used as EKF measurement sources
