import argparse
import time
from pathlib import Path
import numpy as np
from .common import load, save, table, plot
from .core import fit, steering, music, matched_error, blockwise_music
from .data import load_case


def matched_masks(sensors, blocks, schedule):
    """Equal active count and per-sensor uptime; only pair co-observation changes."""
    if blocks % sensors or sensors < 4 or sensors % 2:
        raise ValueError("Matched schedules require even sensors >=4 and blocks divisible by sensors")
    base = np.zeros(sensors, dtype=bool)
    if schedule == "pair_poor":
        base[::2] = True
    elif schedule == "pair_complementary":
        base[:sensors // 2] = True
    else:
        raise ValueError(schedule)
    return np.stack([np.roll(base, b % sensors) for b in range(blocks)])


def prepare(a, c):
    root = Path(a.data)
    root.mkdir(parents=True, exist_ok=True)
    if a.input:
        source = load(a.input)
        records = []
        groups = {}
        for record in source:
            group = str(record.get("group", record["id"]))
            if group in groups and groups[group] != record["split"]:
                raise ValueError("Recording group crosses splits")
            groups[group] = record["split"]
            path = Path(record["path"])
            path = path if path.is_absolute() else Path(a.input).parent / path
            load_case(path)
            records.append(dict(record, path=str(path.resolve())))
    else:
        rng = np.random.default_rng(c["seed"])
        records = []
        positions = np.arange(c["sensors"]) / 2
        schedule = getattr(a, "schedule", "random")
        fixed_masks = None if schedule == "random" else matched_masks(c["sensors"], c["blocks"], schedule)
        for i in range(c["cases"]):
            angles = np.sort(rng.uniform(-55, 55, c["sources"]))
            snr = float(c["snr_db"][i % len(c["snr_db"])])
            blocks = []
            masks = []
            for b in range(c["blocks"]):
                signals = (
                    rng.normal(size=(c["sources"], c["snapshots"]))
                    + 1j * rng.normal(size=(c["sources"], c["snapshots"]))
                ) / np.sqrt(2)
                noise = (
                    (
                        rng.normal(size=(c["sensors"], c["snapshots"]))
                        + 1j * rng.normal(size=(c["sensors"], c["snapshots"]))
                    )
                    / np.sqrt(2)
                    * np.sqrt(c["sources"] * 10 ** (-snr / 10))
                )
                blocks.append(steering(positions, angles) @ signals + noise)
                mask = (rng.random(c["sensors"]) >= c["dropout"]
                        if fixed_masks is None else fixed_masks[b].copy())
                if mask.sum() < 2:
                    mask[:2] = True
                masks.append(mask)
            path = root / f"case_{i:05d}.npz"
            np.savez_compressed(
                path,
                x=np.stack(blocks),
                masks=np.stack(masks),
                positions=positions,
                angles=angles,
            )
            records.append(
                dict(
                    id=str(i),
                    path=str(path.resolve()),
                    split=("train" if i < int(c["cases"] * c.get("training_fraction", 0.))
                           else "val" if i < int(c["cases"] * c.get("training_fraction", 0.)) + max(1, c["cases"] // 5)
                           else "test"),
                    snr_db=snr,
                    source="generated_narrowband_array",
                    schedule=schedule,
                )
            )
    save(root / "manifest.json", records)


def infer(record, c, method, predictor=None):
    data = load_case(record["path"])
    x = data["x"]
    masks = data["masks"]
    positions = data["positions"]
    if method == "tqn_th":
        from . import tqn_th
        return tqn_th.infer(data, c)
    if method == "fcn":
        from . import fcn
        return fcn.infer(data, c, predictor if predictor is not None else fcn.restore(c["fcn_checkpoint"]))
    if method == "full_oracle":
        if not data["full_x"] and not record.get("source", "").startswith(
            "generated"
        ):
            raise ValueError("full_oracle requires full_x")
        full = data["full_x"] if data["full_x"] else x
        covariance = sum(b @ b.conj().T for b in full) / (
            sum(b.shape[1] for b in full)
        )
        angles, grid, spectrum = music(
            covariance, positions, c["sources"], c["grid_step"]
        )
        return dict(
            angles=angles, quality=1.0, lag_coverage=1.0, information_rank=c["sources"]
        )
    if method == "blockwise":
        return blockwise_music(x, masks, positions, c)
    if method == "survivor":
        common = np.all(masks, axis=0)
        if common.sum() <= c["sources"]:
            raise ValueError("Too few common surviving sensors")
        values = np.concatenate([b[common] for b in x], axis=1)
        covariance = values @ values.conj().T / values.shape[1]
        angles, grid, spectrum = music(
            covariance, positions[common], c["sources"], c["grid_step"]
        )
        return dict(
            angles=angles,
            quality=float(common.mean()),
            lag_coverage=float(common.mean()),
            information_rank=c["sources"],
        )
    return fit(x, masks, positions, c, method)


def evaluate(a, c, method, split="test"):
    records = [r for r in load(Path(a.data) / "manifest.json") if r["split"] == split]
    rows = []
    predictor = None
    if method == "fcn":
        from .fcn import restore
        predictor = restore(c["fcn_checkpoint"])
    for record in records:
        start = time.perf_counter()
        try:
            result = infer(record, c, method, predictor)
            errors = matched_error(result["angles"], np.load(record["path"])["angles"])
            error = float(np.sqrt(np.mean(errors**2)))
            mae = float(np.mean(errors))
            maximum = float(max(errors))
            failed = False
        except ValueError as exc:
            if str(exc) not in ("Too few common surviving sensors", "No block has enough surviving sensors"):
                raise
            result = dict(angles=[], quality=0.0, lag_coverage=0.0, information_rank=0)
            error = mae = maximum = 90.0
            failed = True
        elapsed = time.perf_counter() - start
        rows.append(
            dict(
                id=record["id"],
                method=method,
                snr_db=record.get("snr_db"),
                rmse=error,
                mae=mae,
                resolved=maximum <= c["resolution_degrees"],
                catastrophic=maximum > c["catastrophic_degrees"],
                failure=failed or maximum > c["catastrophic_degrees"],
                quality=result["quality"],
                information_rank=result["information_rank"],
                lag_coverage=result["lag_coverage"],
                runtime_per_block_ms=1000 * elapsed / len(load_case(record["path"])["x"]),
            )
        )
        if method == "tqn_th" and "diagnostics" in result:
            diagnostic = result["diagnostics"]
            rows[-1].update(virtual_sensors=result["virtual_sensors"],
                            solver_iterations=diagnostic["iterations"],
                            solver_converged=diagnostic["converged"],
                            projection_converged=diagnostic["projection_converged"],
                            primal_residual=diagnostic["primal_residual"],
                            toeplitz_residual=diagnostic["toeplitz_residual"],
                            minimum_eigenvalue=diagnostic["minimum_eigenvalue"],
                            tqn_alpha=diagnostic["alpha"])
    if not rows:
        raise ValueError("No records for selected split")
    threshold = load(a.calibration)["threshold"] if a.calibration else None
    for row in rows:
        row["accepted"] = threshold is not None and row["quality"] >= threshold
    ordered = sorted(rows, key=lambda r: -r["quality"])
    errors = np.array([r["failure"] for r in ordered])
    summary = dict(
        method=method,
        rmse=np.sqrt(np.mean([r["rmse"] ** 2 for r in rows])),
        mae=np.mean([r["mae"] for r in rows]),
        resolution=np.mean([r["resolved"] for r in rows]),
        catastrophic_error=np.mean([r["catastrophic"] for r in rows]),
        failure_rate=np.mean([r["failure"] for r in rows]),
        aurc=np.mean(np.cumsum(errors) / np.arange(1, len(rows) + 1)),
        quality=np.mean([r["quality"] for r in rows]),
        lag_coverage=np.mean([r["lag_coverage"] for r in rows]),
        runtime_per_block_ms=np.mean([r["runtime_per_block_ms"] for r in rows]),
        coverage=np.mean([r["accepted"] for r in rows])
        if threshold is not None
        else None,
    )
    output = Path(a.output) / method
    table(output / "cases.csv", rows)
    save(output / "metrics.json", summary)
    return summary, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["prepare", "train_fcn", "calibrate", "run", "compare", "plot"]
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--schedule", choices=["random", "pair_poor", "pair_complementary"], default="random")
    p.add_argument("--data", default="data/arrays")
    p.add_argument("--output", default="runs")
    p.add_argument(
        "--method",
        choices=[
            "mscl",
            "diagonal",
            "pair_count",
            "completion",
            "tqn_th",
            "blockwise",
            "fcn",
            "survivor",
            "full_oracle",
        ],
        default="mscl",
    )
    p.add_argument("--calibration")
    p.add_argument("--checkpoint")
    a = p.parse_args()
    c = load(a.config)
    if a.checkpoint:
        c["fcn_checkpoint"] = a.checkpoint
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "train_fcn":
        from .fcn import train
        train(a.data, a.output, c)
    elif a.command == "calibrate":
        _, rows = evaluate(a, c, a.method, "val")
        save(
            Path(a.output) / "calibration.json",
            dict(
                threshold=float(
                    np.quantile([r["quality"] for r in rows], 1 - c["coverage"])
                ),
                coverage=c["coverage"],
            ),
        )
    elif a.command == "run":
        evaluate(a, c, a.method)
    elif a.command == "compare":
        rows = [
            evaluate(a, c, m)[0]
            for m in ["survivor", "blockwise", "pair_count", "diagonal", "completion", "tqn_th", "mscl"] + (["fcn"] if c.get("fcn_checkpoint") else [])
        ]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
