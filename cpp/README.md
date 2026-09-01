# C++ EKF core (Eigen + pybind11)

Port of `qlio.ekf.StochasticCloningEKF` — same state layout
(15 core + 6/clone), FEJ semantics, chi-squared gating, Joseph-form updates.

## Build

```bash
apt-get install libeigen3-dev   # Eigen headers
pip install pybind11
bash cpp/build.sh               # -> cpp/qlio_ekf.cpython-*.so
```

## Parity

Both implementations are driven through an identical randomized 2000-step
schedule (propagate at 200 Hz, clone every 10 steps with marginalisation at 22
clones, gated displacement updates incl. injected outliers, stacked batch
updates), comparing R/v/p/bg/ba, every clone pose, and the full covariance
after every step at `atol=1e-10`:

```bash
PYTHONPATH=. python cpp/parity_check.py   # PASS (fej, legacy, no-gate variants)
```

## Benchmark

```bash
PYTHONPATH=. python cpp/bench.py
```

Measured (single core, 20k propagate steps + clone/update schedule, dim ≤ 147):

| impl | steps/s | speedup |
|------|---------|---------|
| python (numpy) | ~3.9k | 1.0× |
| C++ (Eigen, `-O3 -march=native`) | ~13.2k | **3.3×** |

The gap is moderate because numpy already delegates the large covariance
products to BLAS; the C++ win comes from removing per-call overhead on the
many small (3–15 dim) operations.

Usage from python: `import cpp.qlio_ekf as qlio_ekf` (see `parity_check.py`).
