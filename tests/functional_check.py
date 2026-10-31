"""Small Gaussian-array inference check; no learned model or training is required."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mscl_doa import cli
from mscl_doa.core import fit, matched_error, steering


def main():
    work = ROOT / "runs" / "functional-check"
    work.mkdir(parents=True, exist_ok=True)
    c = json.loads((ROOT / "configs/default.json").read_text())
    c.update(cases=5, sensors=4, blocks=4, snapshots=16, sources=1, grid_step=2.)
    a = SimpleNamespace(input=None, data=str(work / "data"), output=str(work / "calibration"),
                        calibration=None, schedule="pair_complementary")
    cli.prepare(a, c)
    records = json.loads((Path(a.data) / "manifest.json").read_text())
    data = np.load(records[0]["path"])
    original = data["x"].copy()
    a.data = str(work / "data_poor")
    a.schedule = "pair_poor"
    cli.prepare(a, c)
    poor_record = json.loads((Path(a.data) / "manifest.json").read_text())[0]
    np.testing.assert_array_equal(original, np.load(poor_record["path"])["x"])
    a.data = str(work / "data")
    result = fit(data["x"], data["masks"], data["positions"], c)
    np.testing.assert_allclose(result["covariance"], result["covariance"].conj().T, atol=1e-6)
    assert np.linalg.eigvalsh(result["covariance"]).min() > -1e-5
    assert np.isfinite(result["quality"]) and result["information_rank"] >= 0
    _, validation = cli.evaluate(a, c, "mscl", "val")
    calibration = Path(a.output) / "calibration.json"
    threshold = float(np.quantile([r["quality"] for r in validation], 1 - c["coverage"]))
    calibration.write_text(json.dumps(dict(threshold=threshold, coverage=c["coverage"])), encoding="utf-8")
    a.calibration = str(calibration)
    a.output = str(work / "evaluation")
    metrics, _ = cli.evaluate(a, c, "mscl")
    for method in ("pair_count", "diagonal", "completion", "full_oracle", "survivor"):
        cli.evaluate(a, c, method)
    assert all(np.isfinite(v) for k, v in metrics.items() if isinstance(v, (int, float)))
    np.testing.assert_allclose(matched_error([10, -10], [-12, 11]), [1, 2])
    covariance = steering(np.arange(4) / 2, [20])
    assert covariance.shape == (4, 1)
    poor = cli.matched_masks(8, 8, "pair_poor")
    complementary = cli.matched_masks(8, 8, "pair_complementary")
    np.testing.assert_array_equal(poor.sum(0), complementary.sum(0))
    np.testing.assert_array_equal(poor.sum(1), complementary.sum(1))
    assert not np.array_equal(poor.T @ poor.astype(int), complementary.T @ complementary.astype(int))
    print(json.dumps(dict(check="passed", metrics=metrics)))


if __name__ == "__main__":
    main()
