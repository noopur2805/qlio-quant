# QLIO-Quant: measured findings

Multimodal (camera + learned-inertial) stochastic-cloning EKF, studied under
quantization and consistency fixes. Everything below is measured by
`scripts/run_study.py`, `scripts/run_euroc.py`, `cpp/parity_check.py` /
`cpp/bench.py` — commands in the top-level README.

## 1. Filter consistency: FEJ on the learned-displacement update

The displacement measurement is expressed in a gravity-aligned frame anchored
at a stochastic clone. Re-deriving that frame from the evolving clone estimate
at every update (the "obvious" implementation) re-linearises the same clone
tens of times and leaks information into the unobservable directions
(global xy, yaw). Freezing the frame at cloning (first-estimates Jacobians)
makes H constant and closes the leak.

Synthetic 30 s diagnostic (oracle displacement, 20 Hz inflated updates):

| filter | NEES/dof | drift |
|---|---|---|
| estimate-linearised (legacy) | 3.93 | 0.60 % |
| FEJ (frozen clone frame) | **0.35** | **0.13 %** |

On 40 s sequences the same A/B gives NEES 5.7→0.35 and 2.0→0.99 (1.0 =
perfectly calibrated); drift improves up to 25×.

Two second-order findings worth stating honestly:

- **The overlap "double-counting" pathology was mostly a re-linearisation
  artifact.** Without FEJ, fusing 20 Hz overlapping windows as independent
  explodes NEES (175 vs 5.7 with the window/stride covariance inflation).
  Under FEJ the uninflated filter measures NEES 0.76 — already consistent.
  The inflation heuristic mainly compensated for the linearisation leak, not
  for genuine information double-counting (with a real network whose errors
  are correlated across windows, inflation still helps; see §3).
- **Camera-side FEJ did not help and is not enabled.** Both OpenVINS-style
  (Jacobians at first estimates, estimate-triangulated landmarks) and
  fej-triangulated variants were measured: accuracy was a wash but the yaw/xy
  z² leak did not close, because the residual's landmark sensitivity is
  evaluated at the estimate while the nullspace projection is built at the
  first estimates. Closing it needs OC-EKF-style observability constraints —
  documented in `qlio/camera.py`, deliberately out of scope.

## 2. Quantization vs. learned covariance (full TLIO dataset)

Trained on the full TLIO golden split (283 train / 35 val / 36 test sequences,
908k training windows, 1 s @ 200 Hz), small ResNet-1D, MSE→NLL schedule.
fp32 test: **RMSE 0.231 m, mean z² 1.19** (z² = (err/σ)², 1.0 = calibrated).

Per-scope fake-quant sweep (weights + activations at N bits):

| scope | int8 | int6 | int4 |
|---|---|---|---|
| mean head only | z² 1.24 | z² 1.63 | **z² 5.05** (RMSE 0.240) |
| cov head only | z² 1.21 | z² 1.17 | z² 1.37 |
| trunk only | z² 1.06 | z² 0.90 (RMSE 0.364) | z² 2.85 (RMSE 0.650) |
| all | z² 1.06 | z² 0.89 (RMSE 0.356) | z² 3.36 (RMSE 0.647) |

The dangerous failure mode at 4 bits is the **mean head**: displacement RMSE
barely moves (0.231→0.240) while calibration collapses (z² 5) — the filter is
fed confidently wrong measurements and nothing in the accuracy metrics warns
about it. The covariance head itself is surprisingly robust down to 4 bits
once trained on enough data. (On small training subsets the cov head was the
fragile part — the conclusion is data-budget dependent, which is itself a
finding.)

Native int8 static PTQ (fbgemm, real kernels): RMSE 0.252, z² 2.41,
**p50 latency 0.44 ms vs 0.97 ms fp32 (2.2×)**, 60 KB vs 116 KB serialized.
Native quantization touches more of the graph than the per-scope simulation
(quantized add/pool, fused conv-bn), which costs some calibration.

## 3. Quantized networks inside the EKF (real 5-min test sequence)

Filter: FEJ, 20 Hz updates, window/stride covariance inflation. IMU dead
reckoning on this sequence: 1942 m ATE.

| net | ATE | drift | NEES/dof |
|---|---|---|---|
| fp32 | **3.45 m** | **4.25 %** | 2.78 |
| int8 native | 6.28 m | 6.26 % | 6.02 |
| int8 fake-quant (all) | 5.07 m | 6.43 % | 6.28 |
| int8 + σ recalibration | 4.95 m | 6.21 % | 5.57 |
| int8 trunk (cov fp32) | 5.05 m | 6.39 % | 6.57 |

int8 costs ~50 % more drift and roughly doubles NEES on real data. Post-hoc
per-axis σ recalibration (temperature scaling fitted on val) no longer
rescues consistency the way it did in earlier small-data experiments — the
int8 damage on full data is not a pure scale error. NEES > 1 for all
configs reflects temporally correlated network errors across overlapping
windows, which the diagonal-R + inflation model only approximates.

## 4. Real images end-to-end: EuRoC MH_01

KLT front-end (pyramidal LK, forward-backward check, RANSAC F-matrix pruning,
undistortion) on the real MH_01 camera stream, fused by the same MSCKF filter
— no oracle association anywhere. Ground truth is the position-only Leica
track; metrics after standard 4-DoF (yaw+translation) alignment.

135 s flight, ~204 tracked features/frame, 44k features fused:

| method | ATE | drift |
|---|---|---|
| camera-only VIO (real tracker) | **0.33 m** | **0.93 %** |
| IMU dead reckoning | 7561 m | — |

The TLIO-style learned update is intentionally not run here: the network is
trained on pedestrian head-mounted IMU and does not transfer to a drone.

## 5. C++ core

`cpp/qlio_ekf.cpp` (Eigen + pybind11) ports the full filter. A randomized
2000-step lockstep parity check (propagate/clone/marginalise/gated updates,
FEJ and legacy variants) matches state and full covariance at `atol=1e-10`.
Benchmark on the mixed schedule: **3.3×** over numpy (13.2k vs 3.9k steps/s,
single core) — modest because numpy already delegates the large covariance
products to BLAS.

## Known limitations

- Camera-side consistency still leaks in yaw/xy (needs OC-EKF; measured, §1).
- EuRoC attitude "ground truth" is gravity-init + gyro integration (Leica is
  position-only): only position metrics are meaningful there.
- The `small` ResNet is the study workhorse; the full TLIO-size model needs a
  GPU (command in README) — trends above should be re-checked at that scale.
- Displacement-update R is diagonal + scalar inflation; a fitted AR(1)
  correlation model for overlapping windows is the natural next step.
