"""Original-equation regressions and a small TQN-TH/MUSIC evaluation."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mscl_doa import cli, tqn_th
from mscl_doa.core import matched_error, steering


def main():
    c = json.loads((ROOT / "configs/tqn_th.json").read_text())
    # Equation (28) must pair opposite diagonals, including complex conjugation.
    projected = tqn_th.toeplitz_hermitian(np.array([[2 + 1j, 4 + 6j], [2 + 4j, 6 - 1j]]))
    np.testing.assert_allclose(projected, [[4, 3 + 1j], [3 - 1j, 4]])
    np.testing.assert_allclose(tqn_th.positive_projection(np.diag([-1., 2.]), .01), np.diag([.01, 2.]))
    # The large branch is unchanged; the small branch scales instead of soft-thresholding.
    shrunk = tqn_th.tqn_svt(np.diag([2., .5]), 1., .01, 3.)
    np.testing.assert_allclose(shrunk, np.diag([2., .5 / (1 + 2 / 3.01)]))
    known = np.diag([True, True])
    np.testing.assert_allclose(tqn_th.initial_toeplitz(np.diag([0., 4.]), known), 2 * np.eye(2))
    # A full 2x2 covariance has an analytic eigenbasis; both TQN branches are exercised.
    observed = np.array([[4, 1 + 2j], [1 - 2j, 4]])
    completed, diagnostic, _ = tqn_th.complete(observed, np.ones((2, 2), bool), c)
    lower = (4 - np.sqrt(5)) / (1 + .06 / 3.01)
    upper = 4 + np.sqrt(5)
    expected = (lower + upper) / 2 * np.eye(2) + (upper - lower) / (2 * np.sqrt(5)) * (observed - 4 * np.eye(2))
    np.testing.assert_allclose(completed, expected, atol=1e-12)
    assert diagnostic["converged"] and diagnostic["iterations"] == 1
    # Eq. (22) literally uses the zero-filled Rhat even off Omega. This catches
    # an unintended replacement with a newly derived masked proximal update.
    partial, partial_diagnostic, _ = tqn_th.complete(
        np.diag([4., 0., 4.]), np.diag([True, False, True]), dict(c, tqn_max_iterations=1))
    scale = 1 / (1 + .06 / 3.01)
    diagonal = 4 * (2 + 3 / 3.01) / 3 * scale
    np.testing.assert_allclose(partial, np.eye(3) * diagonal, atol=1e-12)
    assert not partial_diagnostic["converged"]
    # Ragged snapshots must weight same-lag observations by their true counts.
    blocks = [np.array([[1., 1.], [2., 2.]]), np.array([[3.], [4.]])]
    pooled, mask, positions, coverage = tqn_th.virtual_observation(blocks, np.ones((2, 2), bool), [0, .5])
    np.testing.assert_allclose(pooled, [[35 / 6, 16 / 3], [16 / 3, 35 / 6]])
    assert mask.all() and coverage == 1.
    for coordinate in ([0., .51], [0., 0.]):
        try:
            tqn_th.virtual_observation(blocks, np.ones((2, 2), bool), coordinate)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid virtual-grid coordinates were accepted")
    # Hide invalid samples: absent sensors must never enter the covariance.
    dirty = [np.array([[1., 2.], [np.nan, np.nan]])]
    hidden, _, _, _ = tqn_th.virtual_observation(dirty, np.array([[True, False]]), [0., .5])
    assert np.isfinite(hidden).all()
    # Orthogonal source waveforms give an exact Toeplitz covariance and known DOAs.
    positions = np.arange(8) / 2
    truth = np.array([-20., 25.])
    sample = np.arange(64)
    signals = np.exp(2j * np.pi * np.array([3, 7])[:, None] * sample / 64)
    values = steering(positions, truth) @ signals
    exact = tqn_th.infer(dict(x=[values], masks=np.ones((1, 8), bool), positions=positions), c)
    np.testing.assert_allclose(matched_error(exact["angles"], truth), 0., atol=1e-10)
    assert exact["information_rank"] is None
    assert exact["diagnostics"]["minimum_eigenvalue"] >= c["tqn_psd_floor"] - 1e-10
    # The original paper's fixed-failure nested geometry has a 20-element virtual ULA.
    rng = np.random.default_rng(25)
    nested_positions = np.array([1, 2, 3, 4, 5, 10, 15, 20]) / 2
    nested_truth = np.array([-20., 10., 50.])
    signals = (rng.normal(size=(3, 500)) + 1j * rng.normal(size=(3, 500))) / np.sqrt(2)
    noise = np.sqrt(3 / 2) * (rng.normal(size=(8, 500)) + 1j * rng.normal(size=(8, 500)))
    nested_values = steering(nested_positions, nested_truth) @ signals + noise
    nested = tqn_th.infer(dict(x=[nested_values], masks=np.array([[1, 0, 1, 0, 1, 0, 1, 1]], bool),
                               positions=nested_positions), dict(c, sources=3, tqn_alpha=.008))
    assert nested["virtual_sensors"] == 20 and np.isfinite(nested["spectrum"]).all()
    assert nested["diagnostics"]["minimum_eigenvalue"] >= c["tqn_psd_floor"] - 1e-10
    # Same existing CLI protocol: held-out validation calibration, saved load, test CSV.
    work = ROOT / "runs" / "tqn-th-check"
    a = SimpleNamespace(input=None, data=str(work / "data"), output=str(work / "calibration"),
                        calibration=None, schedule="random")
    small = dict(c, cases=6, sensors=8, blocks=3, snapshots=32, snr_db=[10], grid_step=1.)
    cli.prepare(a, small)
    _, validation = cli.evaluate(a, small, "tqn_th", "val")
    calibration = Path(a.output) / "calibration.json"
    cli.save(calibration, dict(threshold=float(np.quantile([r["quality"] for r in validation], 1 - c["coverage"])),
                              coverage=c["coverage"]))
    a.calibration = str(calibration)
    a.output = str(work / "evaluation")
    metrics, rows = cli.evaluate(a, small, "tqn_th")
    assert len(rows) == 5 and all(np.isfinite(row["rmse"]) for row in rows)
    assert all(row["information_rank"] is None and row["virtual_sensors"] == 8 for row in rows)
    assert (Path(a.output) / "tqn_th" / "cases.csv").is_file()
    report = dict(check="passed", exact_doa_error_degrees=matched_error(exact["angles"], truth).tolist(),
                  full_matrix_diagnostics=diagnostic, nested_diagnostics=nested["diagnostics"],
                  nested_doa_error_degrees=matched_error(nested["angles"], nested_truth).tolist(),
                  held_out_metrics=metrics,
                  test_cases=len(rows), converged_cases=sum(row["solver_converged"] for row in rows))
    cli.save(work / "validation.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
