# MSCL-DOA

Implementation of mask-sequence coarray likelihood inference and a local DOA reliability score.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`prepare` generates independent complex Gaussian narrowband array cases with half-wavelength sensor spacing and recorded sensor masks. `--input manifest.json` instead accepts `{id,path,split,group}` records whose NPZ files contain complex `x` [block,sensor,snapshot], `masks`, wavelength-unit `positions` and degree-unit `angles`. `group` identifies a recording; it is optional only when every ID is an independent recording. The importer rejects supplied recording groups crossing val/test and checks the array/mask geometry.

```bash
mscl-doa prepare --data data/arrays
```

## Training and execution

`configs/default.json` records seed `0`. Keep it fixed across paired methods and use separate output directories.

```bash
mscl-doa calibrate --data data/arrays --output checkpoints
mscl-doa run --data data/arrays --calibration checkpoints/calibration.json --output runs
```

## Comparisons and ablations

`--method` selects the implemented method or component control. Train separate checkpoints where applicable. `compare` executes the array estimators on the same cases.

```bash
mscl-doa compare --data data/arrays --calibration checkpoints/calibration.json --output runs
mscl-doa plot --output runs
```

## Outputs

The explicit run directories contain `cases.csv`, `metrics.json`. Comparisons write CSV/JSON and plots derived from those saved measurements.

## Verification

Run `python tests/functional_check.py` after installing dependencies. This inference-only seed-0 CPU check creates five generated horizons, solves the constrained complex covariance problem, checks Hermitian/PSD output, calibrates a threshold only on validation horizons, evaluates every available estimator, verifies saved calibration loading, checks angular assignment by hand, and checks exact per-sensor uptime/active-count equality and identical full-array samples for matched schedules. A separate six-sensor two-source case verifies both DOA estimates and rank-two nuisance-profiled information. Passed with Python 3.13.5, numpy 2.1.3, scipy 1.15.3, cvxpy 1.9.3, CLARABEL 0.11.1 and qdldl 0.1.9.post1. In the Windows verification environment the last three packages were installed in an ignored repository-local dependency directory, without modifying shared packages. Temporary outputs are generated under ignored `runs/functional-check/`. This does not establish simulation-table or LOCATA performance. No neural training is required by the MSCL estimator.

## Method and metric entry points

| Paper object | Entry point | Configuration or command |
|---|---|---|
| Mask-specific covariance and proper-complex error precision | `src/mscl_doa/core.py:observations` | `shrinkage`; `x`, `masks` |
| Shared physical-lag WLS with conjugate symmetry and physical-array PSD | `core.fit` | `regularization`, `solver`; `mscl-doa run` |
| Nuisance-profiled local score | `core.reliability` | fixed pseudoinverse `rcond=1e-9`; local eigenmode tolerance `1e-9` |
| RMSE, MAE, resolution, catastrophic/failure rate, AURC, coverage, lag coverage and runtime | `cli.evaluate`, `core.matched_error` | `resolution_degrees`, `catastrophic_degrees`; `cases.csv`, `metrics.json` |
| Pair-count/diagonal weighting, regularization control, survivor and oracle | `cli.infer`, `core.fit` | `--method pair_count`, `diagonal`, `completion`, `survivor`, `full_oracle` |
| Equal-duty pair schedules | `cli.matched_masks`, `cli.prepare` | `--schedule pair_poor` or `pair_complementary` |

For the defining schedule intervention, use the same seed and config in both commands and separate data/output directories:

```bash
mscl-doa prepare --config configs/default.json --schedule pair_poor --data data/pair_poor
mscl-doa prepare --config configs/default.json --schedule pair_complementary --data data/pair_complementary
```

The matched generator requires even sensor count and a block count divisible by the sensor count. It cyclically shifts an alternating mask or contiguous half-array mask; both have identical active count, per-sensor uptime, union sensors and full-array signal/noise draws, but different pair co-observation. `dropout` applies only to `--schedule random`.

`completion` now solves a genuine structured PSD covariance-completion objective with physical-lag conjugate tying and trace/nuclear-norm regularization, rather than multiplying the MSCL ridge by ten. `blockwise` averages snapshot-weighted normalized MUSIC spectra from surviving subarrays. These controls have their own objectives. The local MSCL score remains a conditioning descriptor and is interpreted alongside information rank.

## Variable snapshots, measured audio and FCN

NPZ imports accept dense `x` plus per-block `snapshots`, or numbered `x_0000,x_0001,...` arrays with different snapshot lengths. Matching `full_x_0000,...` arrays support the full-array oracle when physically recorded. No object arrays or pickle are needed. Every covariance/precision uses its own block count; the full-array oracle weights covariances by total snapshots.

`python -m mscl_doa.convert --manifest recordings.json --output data/measured` reads original multichannel WAV and LOCATA tab-delimited `position_array_*.txt` / `position_source_*.txt` tables. Input records contain `id,group,split,audio,channels,array_position,source_positions,duration_seconds,frequency_hz`; `channels` is an explicit zero-based linear subset. Native `mic1_x/mic1_y/mic1_z` fields are read directly from array pose tables, together with `rotation_11` through `rotation_33`; alternatively provide a `microphones` xyz table with `microphone_frame: "world"` or `"array"`. An explicit `array_axis` uses world coordinates unless `axis_frame: "array"` is supplied. Static orientation rotates local axes into world coordinates before direction-cosine angles are computed. Field semantics were checked against the [official LOCATA I/O snapshot](https://github.com/cevers/sap_locata_io/tree/632468f49b13ca5a9f2f51c064d91183c06622fa), distributed with the corpus under ODC-BY 1.0; this reader is independently implemented.

The converter rejects non-linear subsets and moving arrays/sources because one horizon assumes fixed physical lags and angles. It extracts one explicitly declared STFT bin (`n_fft` default 512, half overlap), scales physical spacing to wavelengths at the actual bin frequency, accepts arbitrary `block_snapshots` and controlled `masks`, and preserves independent recording groups across calibration/test. Task-1/2 static LOCATA development recordings provide the required source annotations; no hidden evaluation labels are invented. These are controlled dropouts applied to measured waveforms. The exported angle is the linear-array direction cosine in degrees, distinct from the full LOCATA azimuth/elevation tracking task.

```bash
python -m mscl_doa.convert --manifest recordings.json --output data/measured
mscl-doa prepare --input data/measured/manifest.json --data data/arrays
mscl-doa run --method blockwise --data data/arrays --output runs/blockwise
```

`src/mscl_doa/fcn.py` implements the time-varying-array proxy-spectrum FCN described by [Dai et al., IEICE 2025](https://globals.ieice.org/en_transactions/communications/10.23919/transcom.2024EBP3103/_f): jointly observed off-diagonal covariance proxy B^H r; real/imaginary/phase channels; length-preserving kernel-31 convolution/BN/ReLU; shallow/deep concatenation skips; sigmoid grid classification trained with BCE, Adam and OneCycleLR. This is a clean implementation with explicit configurable widths, rather than author-provided pretrained code. `fcn_width` defaults to 64; the exact published parameter count is not asserted.

```bash
mscl-doa prepare --config configs/fcn.json --data data/fcn
mscl-doa train_fcn --config configs/fcn.json --data data/fcn --output checkpoints/fcn
mscl-doa run --config configs/fcn.json --method fcn --checkpoint checkpoints/fcn/fcn.pt --data data/fcn --output runs/fcn
```

The supplied FCN config generates independent labelled training horizons before held-out validation/test and sets the paper's primary 12 sensors. Its training split is explicit (`training_fraction`); training refuses validation/test-only data. The FCN defaults to 200 epochs for real experiments; the functional check overrides one epoch and width eight. Network loading occurs before measured inference runtime.

`python tests/protocol_check.py` passed original generated multichannel WAV and native LOCATA pose/source TSV conversion with a 90-degree array rotation, unequal block snapshots [6,9,10], MSCL/completion/blockwise/full-oracle angular errors at most one degree, and actual FCN training, checkpoint reload and held-out evaluation. `tests/functional_check.py` also passed PSD, nuisance-rank and equal-duty schedule regressions. No full FCN training or LOCATA benchmark experiment was run.

## Published TQN-TH structured completion

`src/mscl_doa/tqn_th.py` independently implements [Wu et al., *Robust DOA Estimation for Sparse Arrays under Sensor Failures via Structured Matrix Completion* (2025)](https://link.springer.com/article/10.1007/s11760-025-04544-3), DOI `10.1007/s11760-025-04544-3`. The full original HTML, Equations (16)–(34), Section 4 parameters and the complete [Algorithm 1 image](https://link.springer.com/article/10.1007/s11760-025-04544-3/figures/a) were read through legitimate East China Normal University institutional access on 2026-11-01. This implementation does not depend on author code or redistribute the article.

The objective uses the truncated quadratic nuclear penalty `sum(min(alpha * sigma_i**2, 1))`. Each iteration executes the printed intermediate update (22), the piecewise TQN-SVT (24), alternating paired-diagonal Toeplitz/Hermitian projection (28) and eigenvalue-floor PSD projection (30), the multiplier update (33), and the stated relative stopping tests. Known entries, including known zero values, initialize each lag; missing lag entries initialize to zero. `completion` remains the separate trace-regularized baseline.

Two details of the original equations are preserved explicitly. Equation (17) defines `tau=sqrt(1/alpha)`, whereas a Section 4 sentence says `tau=1/alpha`; the implementation follows Equation (17). Equation (22) uses the zero-filled observation matrix across the whole update, including outside the observation mask. The code implements that printed expression literally and does not claim that this expression is an exact minimizer of the preceding masked subproblem. The original PSD floor uses the epsilon symbol also used for the outer tolerance; `tqn_psd_floor` defaults to that tolerance and can be declared separately.

`configs/tqn_th.json` fixes the Section 4 settings `mu=.01`, `rho=3`, maximum 200 outer iterations, outer tolerance `1e-4`, maximum 10 projection iterations, projection tolerance `1e-6`, numerical denominator epsilon `1e-8`, and `alpha=.03`. The paper additionally uses `alpha=.008` at low SNR. Select and freeze alpha in the experiment configuration before evaluation; it is never chosen from test errors or hidden test labels. The supplied config covers the main paper's 12-sensor, 8-block uniform-array experiment. It does not recreate Wu's simulation sweep automatically.

Wu's original input has a fixed incomplete coarray. The mask-sequence adapter pools only actually observed sensor-pair covariances at the same physical lag, weighted by each block's snapshot count, before running the original algorithm on one incomplete virtual ULA covariance. Physical positions must lie on the explicitly declared uniform grid (`tqn_grid_spacing=.5` wavelength by default); sparse gaps and a translation of the array are supported. Coordinate-perturbed or measured off-grid geometries are rejected rather than silently snapped. The virtual ULA has one element per grid coordinate over the full physical aperture: Wu's 8-sensor nested array has 20 virtual elements. The repository's common known-source-count MUSIC stage then operates on those virtual positions.

```bash
mscl-doa prepare --config configs/tqn_th.json --data data/tqn
mscl-doa calibrate --config configs/tqn_th.json --method tqn_th --data data/tqn --output checkpoints/tqn
mscl-doa run --config configs/tqn_th.json --method tqn_th --data data/tqn --calibration checkpoints/tqn/calibration.json --output runs/tqn
python tests/tqn_th_check.py
```

The original method has no selective certificate. Its `quality` here is the named normalized MUSIC eigengap heuristic, calibrated on validation horizons, and its `information_rank` is blank rather than an invented MSCL information rank. Per-case CSV records virtual array size, outer iterations, both convergence flags, primal residual, Toeplitz residual, minimum eigenvalue and alpha. Reaching an iteration cap is reported, including when the last PSD projection still has a nonzero Toeplitz residual.

`tests/tqn_th_check.py` passed on CPU with one numerical thread: hand-checkable complex diagonal averaging and PSD clipping; both branches of TQN-SVT; known-zero initialization; the literal Equation (22) outside the observation mask; snapshot-weighted ragged pooling; exclusion of hidden invalid sensor samples; invalid-grid rejection; and analytic two-source MUSIC errors `[0,0]` degrees. Held-out calibration and CSV/JSON evaluation on six independently generated horizons (one validation, five test) gave test RMSE `0.2923839280` degrees, resolution `1.0`, validation-threshold coverage `0.6`, and five of five converged outer solves. The original MSCL functional check also passed after this integration. These are functionality checks, not a benchmark estimate.

The original nested-array geometry was additionally exercised with five surviving sensors at positions `[1,3,5,15,20]*d`, three sources `[-20,10,50]` degrees and 500 snapshots. This fixture uses unit-power independent sources, noise variance three and the repository's 0 dB total-signal-power SNR convention; the source article does not specify its SNR normalization. The literal published updates converged in four iterations with primal residual `8.16e-6`, Toeplitz residual `2.37e-7` and minimum eigenvalue `1e-4`; the three matched errors were `[0.5,23.25,0.25]` degrees. Thus this check verifies the original geometry and finite constrained output but does not reproduce the paper's reported resolution performance. No full simulation sweep or LOCATA benchmark was run.
