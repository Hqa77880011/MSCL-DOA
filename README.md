# MSCL-DOA

Implementation of mask-sequence coarray likelihood inference and a local DOA reliability score.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Data

`mscl-doa prepare` generates independent complex Gaussian narrowband two-source array cases with recorded sensor masks. The default ULA uses half-wavelength spacing. To import array observations, provide a JSON list with `id`, `path`, `split` (val/test), optional `snr_db` and `source`; NPZ files contain complex `x` [block,sensor,snapshot], boolean `masks` [block,sensor], `positions` in wavelengths and reference `angles` in degrees. Inactive entries are never used. Imported files should store zeros there. Optional `full_x` enables the full-array oracle. Unequal block lengths can be represented by separate jobs with the current fixed-length archive format.

[LOCATA](https://www.locata.lms.tf.fau.de/) recordings must first be converted to narrowband STFT snapshots with array coordinates scaled by frequency/sound speed. Keep recordings and source trajectories in one split. This adapter boundary avoids treating a broadband waveform as a narrowband covariance. The core `fit` function uses actual physical lags and never imposes Toeplitz structure on a nonuniform array.

```bash
mscl-doa prepare --data data/arrays
mscl-doa calibrate --data data/arrays --output checkpoints
mscl-doa run --data data/arrays --calibration checkpoints/calibration.json --output runs
mscl-doa compare --data data/arrays --calibration checkpoints/calibration.json --output runs
mscl-doa plot --output runs
```

## Likelihood and controls

Each block supplies its available covariance entries and a shrinkage estimate of the proper-complex Gaussian covariance law `Omega = R ⊗ conj(R)`, in row-major vector order. Snapshot-weighted correlated WLS is solved over complex physical-lag variables with conjugate symmetry and a PSD covariance on the physical array. A shared MUSIC grid uses the declared source count. Source powers/noise are fitted by nonnegative least squares; nuisance-profiled information supplies the minimum-positive-eigenvalue/mean-trace score. Information rank is reported alongside that relative-conditioning score.

Controls are common-survivor MUSIC, pair-count weighting, diagonal block precision, and a stronger-regularized structured completion objective. `full_oracle` requires true full observations. The pair-count and diagonal controls preserve the same PSD solve and MUSIC stage, isolating precision use. There is no neural FCN training stage in this estimator; `calibrate` freezes a score threshold using validation quality quantiles and the desired coverage.

## Settings and results

`configs/default.json` controls simulation seed, sensors, masks, blocks, snapshots, SNR grid, source count, shrinkage, regularization, MUSIC resolution, failure/resolution tolerances and CVXPY solver. CLARABEL is the default PSD-capable solver. `run` writes per-case `cases.csv` and aggregate `metrics.json`. Metrics include permutation-matched RMSE/MAE, resolution probability, catastrophic/failure rate, lag coverage, information quality/rank, AURC, calibrated coverage and measured runtime per block. A survivor run with too few sensors is an explicit failure assigned 90-degree error. Solver failures remain errors.

Use identical `x` and altered mask archives for repeated/complementary same-duty controls, and identical masks with altered block noise/snapshot count for precision controls. Aggregate rates use [0,1]. Per-SNR results can be read from `cases.csv`; overall RMSE pools squared angular errors rather than averaging case RMSE. Comparison CSV/JSON and PNG/SVG charts use measured values.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
mscl-doa-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.
