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

Run `python tests/functional_check.py` after installing dependencies. This inference-only seed-0 CPU check creates five generated horizons, solves the constrained complex covariance problem, checks Hermitian/PSD output, calibrates a threshold only on validation horizons, evaluates every available estimator, verifies saved calibration loading, checks angular assignment by hand, and checks exact per-sensor uptime/active-count equality of matched schedules. Passed with Python 3.13.5, numpy 2.1.3, scipy 1.15.3, cvxpy 1.9.3, CLARABEL 0.11.1 and qdldl 0.1.9.post1. In the Windows verification environment the last three packages were installed in an ignored repository-local dependency directory, without modifying shared packages. Outputs remain under ignored `runs/functional-check/`. This does not establish simulation-table or LOCATA performance. No neural training is required by the MSCL estimator.

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

The current compact controls are declared implementation choices. `completion` is the same PSD lag estimator with tenfold regularization, not the paper's external structured-completion algorithm. The time-varying FCN, blockwise MUSIC baseline and archived paper experiment code are not included. Dense NPZ inputs currently imply equal snapshot count per block; variable-length blocks require a data adapter. LOCATA acquisition, STFT/narrowband extraction and measured array/DOA alignment are external preprocessing steps. No natural-dropout claim is made. The synthetic default uses eight sensors rather than the paper's primary 12; change the config explicitly for that experiment and keep matched schedules' divisibility requirement. The local score is a relative conditioning descriptor and its information rank must be interpreted alongside it.
