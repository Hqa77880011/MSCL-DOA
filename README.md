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

`prepare` generates independent complex Gaussian narrowband array cases with half-wavelength sensor spacing and recorded sensor masks. `--input manifest.json` instead accepts `{id,path,split}` records whose NPZ files contain complex `x` [block,sensor,snapshot], `masks`, wavelength-unit `positions` and degree-unit `angles`; preserve recording groups across val/test.

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

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
