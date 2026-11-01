# MSCL-DOA

Mask-sequence coarray likelihood inference and a local DOA reliability score.

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

## Method and metric entry points

| Component | Entry point | Configuration or command |
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

`completion` solves structured PSD covariance completion with conjugate physical-lag tying and trace/nuclear-norm regularization. `blockwise` averages snapshot-weighted normalized MUSIC spectra from surviving subarrays. Interpret the MSCL conditioning score alongside information rank.

## Variable snapshots, measured audio and FCN

NPZ imports accept dense `x` plus per-block `snapshots`, or numbered `x_0000,x_0001,...` arrays with different snapshot lengths. Matching `full_x_0000,...` arrays support the full-array oracle when physically recorded. No object arrays or pickle are needed. Every covariance/precision uses its own block count; the full-array oracle weights covariances by total snapshots.

`python -m mscl_doa.convert --manifest recordings.json --output data/measured` reads original multichannel WAV and LOCATA tab-delimited `position_array_*.txt` / `position_source_*.txt` tables. Input records contain `id,group,split,audio,channels,array_position,source_positions,duration_seconds,frequency_hz`; `channels` is an explicit zero-based linear subset. Native `mic1_x/mic1_y/mic1_z` fields are read directly from array pose tables, together with `rotation_11` through `rotation_33`; alternatively provide a `microphones` xyz table with `microphone_frame: "world"` or `"array"`. An explicit `array_axis` uses world coordinates unless `axis_frame: "array"` is supplied. Static orientation rotates local axes into world coordinates before direction-cosine angles are computed. Field semantics were checked against the [official LOCATA I/O snapshot](https://github.com/cevers/sap_locata_io/tree/632468f49b13ca5a9f2f51c064d91183c06622fa), distributed with the corpus under ODC-BY 1.0; this reader is independently implemented.

The converter rejects non-linear subsets and moving arrays/sources because one horizon assumes fixed physical lags and angles. It extracts one explicitly declared STFT bin (`n_fft` default 512, half overlap), scales physical spacing to wavelengths at the actual bin frequency, accepts arbitrary `block_snapshots` and controlled `masks`, and preserves independent recording groups across calibration/test. Task-1/2 static LOCATA development recordings provide the required source annotations; no hidden evaluation labels are invented. These are controlled dropouts applied to measured waveforms. The exported angle is the linear-array direction cosine in degrees, distinct from the full LOCATA azimuth/elevation tracking task.

```bash
python -m mscl_doa.convert --manifest recordings.json --output data/measured
mscl-doa prepare --input data/measured/manifest.json --data data/arrays
mscl-doa run --method blockwise --data data/arrays --output runs/blockwise
```

`src/mscl_doa/fcn.py` implements the proxy-spectrum FCN from [Dai et al., IEICE 2025](https://globals.ieice.org/en_transactions/communications/10.23919/transcom.2024EBP3103/_f): jointly observed off-diagonal covariance proxy B^H r; real/imaginary/phase channels; kernel-31 convolution/BN/ReLU; shallow/deep skips; and sigmoid grid classification with BCE, Adam and OneCycleLR. `fcn_width` defaults to 64.

```bash
mscl-doa prepare --config configs/fcn.json --data data/fcn
mscl-doa train_fcn --config configs/fcn.json --data data/fcn --output checkpoints/fcn
mscl-doa run --config configs/fcn.json --method fcn --checkpoint checkpoints/fcn/fcn.pt --data data/fcn --output runs/fcn
```

The FCN config uses 12 sensors and independent labelled training horizons before validation/test. `training_fraction` controls the training split; `epochs` defaults to 200. Network loading precedes inference timing.

## TQN-TH structured completion

`src/mscl_doa/tqn_th.py` implements truncated quadratic nuclear-norm completion from [Wu et al., Robust DOA Estimation for Sparse Arrays under Sensor Failures via Structured Matrix Completion](https://link.springer.com/article/10.1007/s11760-025-04544-3), DOI `10.1007/s11760-025-04544-3`.

The objective uses the truncated quadratic nuclear penalty `sum(min(alpha * sigma_i**2, 1))`. Each iteration executes the printed intermediate update (22), the piecewise TQN-SVT (24), alternating paired-diagonal Toeplitz/Hermitian projection (28) and eigenvalue-floor PSD projection (30), the multiplier update (33), and the stated relative stopping tests. Known entries, including known zero values, initialize each lag; missing lag entries initialize to zero. `completion` remains the separate trace-regularized baseline.

The SVT threshold is `tau=sqrt(1/alpha)`. The intermediate update uses the zero-filled observation matrix over the whole matrix, including entries outside the mask. `tqn_psd_floor` defaults to the outer tolerance and can be configured separately.

`configs/tqn_th.json` sets `mu=.01`, `rho=3`, 200 outer iterations, outer tolerance `1e-4`, 10 projection iterations, projection tolerance `1e-6`, denominator epsilon `1e-8` and `alpha=.03`. Use a prespecified configuration to select alpha, including `.008` for low-SNR settings, before evaluation. The default array has 12 sensors and 8 blocks.

The mask-sequence adapter pools observed sensor-pair covariances by physical lag, weighted by block snapshot count, into an incomplete virtual ULA covariance. Positions must lie on the configured grid (`tqn_grid_spacing=.5` wavelength by default); sparse gaps and translations are supported. The virtual array spans the full aperture, with one element per grid coordinate. Known-source-count MUSIC operates on those virtual positions.

```bash
mscl-doa prepare --config configs/tqn_th.json --data data/tqn
mscl-doa calibrate --config configs/tqn_th.json --method tqn_th --data data/tqn --output checkpoints/tqn
mscl-doa run --config configs/tqn_th.json --method tqn_th --data data/tqn --calibration checkpoints/tqn/calibration.json --output runs/tqn
python tests/tqn_th_check.py
```

TQN-TH `quality` is a normalized MUSIC eigengap calibrated on validation horizons; `information_rank` is blank. Per-case CSV records virtual array size, outer iterations, convergence flags, primal/Toeplitz residuals, minimum eigenvalue and alpha, including iteration-cap outcomes.

## Checks

Generated fixtures check data handling, metrics and checkpoint loading. On Windows with separate MKL/OpenMP runtimes, set `$env:MKL_THREADING_LAYER='SEQUENTIAL'` in the active PowerShell process.

```bash
python tests/functional_check.py
python tests/protocol_check.py
python tests/tqn_th_check.py
```
