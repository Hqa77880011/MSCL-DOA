"""Convert multichannel LOCATA-style WAV and physical geometry to narrowband cases."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from .data import save_case


def table_rows(path):
    """Read LOCATA's original named TSV fields or an explicit CSV geometry table."""
    text = Path(path).read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t;")
    return list(csv.DictReader(text.splitlines(), dialect=dialect))


def xyz_table(path):
    """Read global x/y/z positions from the original LOCATA position tables."""
    rows = table_rows(path)
    return np.asarray([[float(row[k]) for k in ("x", "y", "z")] for row in rows])


def convert(manifest, output):
    """Use an explicit linear microphone subset and static source/array coordinates."""
    records = json.loads(Path(manifest).read_text(encoding="utf-8"))
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    owners, exported = {}, []
    for record in records:
        def path(key):
            value = Path(record[key])
            return value if value.is_absolute() else Path(manifest).parent / value
        group = str(record.get("group", record["audio"]))
        if group in owners and owners[group] != record["split"]:
            raise ValueError("Measured recording crosses calibration/test splits")
        owners[group] = record["split"]
        rate, audio = wavfile.read(path("audio"))
        if audio.ndim != 2:
            raise ValueError("Expected multichannel LOCATA WAV")
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(float) / max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        channels = np.asarray(record["channels"], int)
        pose = table_rows(path("array_position"))
        array = np.asarray([[float(r[k]) for k in ("x", "y", "z")] for r in pose])
        tolerance = record.get("static_tolerance_m", .05)
        if np.max(np.linalg.norm(array - array.mean(0), axis=1)) > tolerance:
            raise ValueError("A shared-lag horizon requires a static array")
        rotation = np.eye(3)
        if "rotation_11" in pose[0]:
            rotations = np.array([[[float(r[f"rotation_{i}{j}"]) for j in (1, 2, 3)] for i in (1, 2, 3)] for r in pose])
            if np.max(abs(rotations - rotations.mean(0))) > record.get("rotation_tolerance", .001):
                raise ValueError("A shared-lag horizon requires a static array orientation")
            rotation = rotations.mean(0)
        if "microphones" in record:
            microphones = xyz_table(path("microphones"))
            if record.get("microphone_frame", "world") == "array":
                microphones = microphones @ rotation.T + array.mean(0)
        else:
            microphone_series = np.array([[[float(r[f"mic{ch + 1}_{k}"]) for k in ("x", "y", "z")] for ch in channels] for r in pose])
            microphones = microphone_series.mean(0)
            if np.max(np.linalg.norm(microphone_series - microphones, axis=2)) > tolerance:
                raise ValueError("Static LOCATA conversion requires stable physical microphones")
        if len(microphones) != len(channels):
            raise ValueError("Provide one physical microphone position per selected channel")
        centered = microphones - microphones.mean(0)
        axis = np.asarray(record.get("array_axis", np.linalg.svd(centered)[2][0]), float)
        if "array_axis" in record and record.get("axis_frame", "world") == "array":
            axis = rotation @ axis
        if "array_axis" not in record and axis @ rotation[:, 0] < 0:
            axis = -axis
        axis /= np.linalg.norm(axis)
        distance = centered @ axis
        if np.max(np.linalg.norm(centered - distance[:, None] * axis, axis=1)) > record.get("linear_tolerance_m", .001):
            raise ValueError("MSCL's linear-array model requires a linear microphone subset")
        start = int(round(record.get("start_seconds", 0.) * rate))
        stop = start + int(round(record["duration_seconds"] * rate))
        if stop > len(audio):
            raise ValueError("Requested horizon exceeds recording duration")
        fft = int(record.get("n_fft", 512))
        frequencies, _, spectra = stft(audio[start:stop, channels].T, fs=rate,
                                      nperseg=fft, noverlap=fft // 2, boundary=None, padded=False)
        index = int(np.argmin(abs(frequencies - record["frequency_hz"])))
        frequency = float(frequencies[index])
        if frequency <= 0:
            raise ValueError("Use a positive narrowband frequency")
        narrow = spectra[:, index]
        counts = record.get("block_snapshots")
        if counts is None:
            boundaries = np.linspace(0, narrow.shape[1], int(record.get("blocks", 8)) + 1).astype(int)
            counts = np.diff(boundaries).tolist()
        else:
            boundaries = np.r_[0, np.cumsum(counts)]
        if min(counts) < 1 or boundaries[-1] > narrow.shape[1]:
            raise ValueError("Insufficient STFT snapshots for the requested blocks")
        blocks = [narrow[:, left:right] for left, right in zip(boundaries[:-1], boundaries[1:])]
        masks = np.asarray(record.get("masks", np.ones((len(blocks), len(channels)), bool)), bool)
        source_files = record["source_positions"]
        directions = []
        for name in source_files:
            p = Path(name); p = p if p.is_absolute() else Path(manifest).parent / p
            source = xyz_table(p)
            if np.max(np.linalg.norm(source - source.mean(0), axis=1)) > record.get("static_tolerance_m", .05):
                raise ValueError("A shared-lag horizon requires static sources")
            delta = source.mean(0) - array.mean(0)
            directions.append(np.rad2deg(np.arcsin(np.clip(delta @ axis / np.linalg.norm(delta), -1, 1))))
        filename = str(record["id"]) + ".npz"
        save_case(root / filename, blocks, masks, distance * frequency / record.get("sound_speed_m_s", 343.),
                  np.asarray(directions), full_blocks=blocks)
        exported.append(dict(id=str(record["id"]), path=filename, split=record["split"], group=group,
                             source="measured_waveform_controlled_mask", frequency_hz=frequency))
    (root / "manifest.json").write_text(json.dumps(exported, indent=2), encoding="utf-8")
    return exported


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); print(len(convert(a.manifest, a.output)))


if __name__ == "__main__":
    main()
