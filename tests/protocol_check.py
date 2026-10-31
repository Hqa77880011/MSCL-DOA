"""Original WAV/position-table conversion and unequal-snapshot inference regression."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.io import wavfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mscl_doa import cli
from mscl_doa.convert import convert
from mscl_doa.core import observations, matched_error
from mscl_doa.data import load_case
from mscl_doa import fcn


def main():
    torch.set_num_threads(1)
    work = ROOT / "runs" / "protocol-check"; work.mkdir(parents=True, exist_ok=True)
    def position(name, values):
        with (work / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t"); writer.writerow(["x", "y", "z"]); writer.writerows(values)
    spacing = 343. / 1000 / 2
    coordinate = (np.arange(6) - 2.5) * spacing
    position("microphones.txt", np.c_[coordinate, np.zeros((6, 2))])
    with (work / "array.txt").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["x", "y", "z"] + [f"rotation_{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)] + [f"mic{ch}_{k}" for ch in range(1, 7) for k in ("x", "y", "z")])
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        microphones = np.c_[np.zeros(6), coordinate, np.zeros(6)]
        writer.writerow([0, 0, 0] + rotation.ravel().tolist() + microphones.ravel().tolist())
    angle = np.deg2rad(20)
    position("source.txt", [[-10 * np.cos(angle), 10 * np.sin(angle), 0]])
    records = []
    rng = np.random.default_rng(17)
    for i, split in enumerate(["val", "test"]):
        time = np.arange(16000) / 16000
        audio = .1 * np.sin(2 * np.pi * 1000 * (time[:, None] + coordinate[None] * np.sin(angle) / 343.))
        audio += rng.normal(0, .0001, audio.shape)
        filename = f"audio_array_{i}.wav"
        wavfile.write(work / filename, 16000, (audio * 32767).astype(np.int16))
        records.append(dict(id=str(i), group=str(i), split=split, audio=filename,
                            channels=list(range(6)), array_position="array.txt",
                            source_positions=["source.txt"], array_axis=[1., 0., 0.], axis_frame="array", duration_seconds=1.,
                            frequency_hz=1000., block_snapshots=[6, 9, 10]))
    manifest = work / "recordings.json"; manifest.write_text(json.dumps(records), encoding="utf-8")
    convert(manifest, work / "converted")
    a = SimpleNamespace(input=str(work / "converted/manifest.json"), data=str(work / "data"),
                        output=str(work / "evaluation"), calibration=None)
    c = json.loads((ROOT / "configs/default.json").read_text()); c.update(sources=1, grid_step=1.)
    cli.prepare(a, c)
    record = json.loads((Path(a.data) / "manifest.json").read_text())[1]
    case = load_case(record["path"])
    obs = observations(case["x"], case["masks"], c)
    assert [r[3] for r in obs] == [6, 9, 10]
    for method in ("mscl", "completion", "blockwise", "full_oracle"):
        result = cli.infer(record, c, method)
        assert matched_error(result["angles"], [20.]).max() <= 1.
        cli.evaluate(a, c, method)
        if "covariance" in result:
            assert np.linalg.eigvalsh(result["covariance"]).min() > -1e-5
    training = SimpleNamespace(input=None, data=str(work / "fcn-data"), schedule="random")
    neural_c = dict(c, cases=10, sensors=6, blocks=4, snapshots=12, training_fraction=.6,
                    fcn_epochs=1, fcn_width=8, fcn_batch_size=2)
    cli.prepare(training, neural_c)
    fcn.train(training.data, work / "fcn-model", neural_c)
    predictor = fcn.restore(work / "fcn-model/fcn.pt")
    manifest = json.loads((Path(training.data) / "manifest.json").read_text())
    heldout = next(r for r in manifest if r["split"] == "test")
    result = cli.infer(heldout, neural_c, "fcn", predictor)
    assert len(result["angles"]) == 1 and np.isfinite(result["spectrum"]).all()
    restored = fcn.restore(work / "fcn-model/fcn.pt")
    np.testing.assert_allclose(result["spectrum"], cli.infer(heldout, neural_c, "fcn", restored)["spectrum"])
    neural_c["fcn_checkpoint"] = str(work / "fcn-model/fcn.pt")
    training.output = str(work / "fcn-evaluation"); training.calibration = None
    cli.evaluate(training, neural_c, "fcn")
    print(json.dumps(dict(check="passed", snapshots=[r[3] for r in obs], original_wav_cases=2, doa_degrees=20., fcn_trained=True)))


if __name__ == "__main__":
    main()
