"""Portable array cases with unequal snapshot counts and no object/pickle arrays."""
from pathlib import Path
import numpy as np


def load_case(path):
    """Load dense ``x`` or numbered ``x_0000`` arrays into a block list."""
    with np.load(path) as raw:
        if "x" in raw:
            lengths = raw["snapshots"] if "snapshots" in raw else [raw["x"].shape[-1]] * len(raw["x"])
            blocks = [block[:, :int(length)] for block, length in zip(raw["x"], lengths)]
        else:
            keys = sorted(k for k in raw.files if k.startswith("x_") and k[2:].isdigit())
            blocks = [raw[k].copy() for k in keys]
        if not blocks:
            raise ValueError("Case has no snapshot blocks")
        masks, positions = raw["masks"].astype(bool), raw["positions"].copy()
        if masks.shape != (len(blocks), len(positions)):
            raise ValueError("Block masks do not match array geometry")
        if any(block.ndim != 2 or block.shape[0] != len(positions) or block.shape[1] < 1 for block in blocks):
            raise ValueError("Each block must be [sensor,positive snapshot count]")
        angles = raw["angles"].copy()
        full = [raw[k].copy() for k in sorted(raw.files) if k.startswith("full_x_")]
        if "full_x" in raw:
            full = list(raw["full_x"])
    return dict(x=blocks, masks=masks, positions=positions, angles=angles, full_x=full)


def save_case(path, blocks, masks, positions, angles, full_blocks=None):
    """Save variable-length blocks using explicit numeric NPZ members."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {f"x_{i:04d}": block for i, block in enumerate(blocks)}
    values.update(masks=masks, positions=positions, angles=angles)
    if full_blocks is not None:
        values.update({f"full_x_{i:04d}": block for i, block in enumerate(full_blocks)})
    np.savez_compressed(path, **values)
