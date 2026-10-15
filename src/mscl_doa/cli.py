import argparse
import time
from pathlib import Path
import numpy as np
from .common import load, save, table, plot
from .core import fit, steering, music, matched_error


def prepare(a, c):
    root = Path(a.data)
    root.mkdir(parents=True, exist_ok=True)
    if a.input:
        source = load(a.input)
        records = []
        for record in source:
            path = Path(record["path"])
            path = path if path.is_absolute() else Path(a.input).parent / path
            records.append(dict(record, path=str(path.resolve())))
    else:
        rng = np.random.default_rng(c["seed"])
        records = []
        positions = np.arange(c["sensors"]) / 2
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
                mask = rng.random(c["sensors"]) >= c["dropout"]
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
                    split="val" if i < c["cases"] // 5 else "test",
                    snr_db=snr,
                    source="generated_narrowband_array",
                )
            )
    save(root / "manifest.json", records)


def infer(record, c, method):
    data = np.load(record["path"])
    x = data["x"]
    masks = data["masks"]
    positions = data["positions"]
    if method == "full_oracle":
        if "full_x" not in data and not record.get("source", "").startswith(
            "generated"
        ):
            raise ValueError("full_oracle requires full_x")
        full = data["full_x"] if "full_x" in data else x
        covariance = sum(b @ b.conj().T for b in full) / (
            full.shape[0] * full.shape[-1]
        )
        angles, grid, spectrum = music(
            covariance, positions, c["sources"], c["grid_step"]
        )
        return dict(
            angles=angles, quality=1.0, lag_coverage=1.0, information_rank=c["sources"]
        )
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
    for record in records:
        start = time.perf_counter()
        try:
            result = infer(record, c, method)
            errors = matched_error(result["angles"], np.load(record["path"])["angles"])
            error = float(np.sqrt(np.mean(errors**2)))
            mae = float(np.mean(errors))
            maximum = float(max(errors))
            failed = False
        except ValueError as exc:
            if str(exc) != "Too few common surviving sensors":
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
                runtime_per_block_ms=1000 * elapsed / len(np.load(record["path"])["x"]),
            )
        )
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
        "command", choices=["prepare", "calibrate", "run", "compare", "plot"]
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--data", default="data/arrays")
    p.add_argument("--output", default="runs")
    p.add_argument(
        "--method",
        choices=[
            "mscl",
            "diagonal",
            "pair_count",
            "completion",
            "survivor",
            "full_oracle",
        ],
        default="mscl",
    )
    p.add_argument("--calibration")
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
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
            for m in ["survivor", "pair_count", "diagonal", "completion", "mscl"]
        ]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
