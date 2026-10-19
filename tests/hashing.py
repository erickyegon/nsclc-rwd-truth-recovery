"""Platform-independent content hash of the generator output (used by test_phase1.py and when recording the expected value).

Why not a plain byte hash: pandas writes CSV with the operating system's line ending, and transcendental functions (log, exp,
power) can differ in the last bit between Windows and Linux. That moves the nanosecond part of the `death_true` timestamp in
truth_patients.csv while every date-level output stays identical (checked file by file: 7 of 8 files were byte-identical across
Windows and Linux CI after line-ending normalisation). So raw tables are hashed as LF-normalised bytes and truth_patients.csv is
hashed after reducing timestamps to calendar days and floats to 6 decimals.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

CRLF, LF = bytes([13, 10]), bytes([10])


def _canonical_truth_patients(path: Path) -> bytes:
    df = pd.read_csv(path)
    for c in df.columns:
        if c in ("death_true", "end_obs"):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.floor("D").dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(6)
    return df.to_csv(index=False, lineterminator="\n").encode()


def content_hash(raw_dir: Path, truth_dir: Path):
    h = hashlib.sha256()
    per_file = {}
    files = sorted(Path(raw_dir).glob("*.csv")) + [Path(truth_dir) / "truth_patients.csv", Path(truth_dir) / "truth_lines.csv"]
    for f in files:
        content = _canonical_truth_patients(f) if f.name == "truth_patients.csv" else f.read_bytes().replace(CRLF, LF)
        per_file[f.name] = hashlib.sha256(content).hexdigest()[:12]
        h.update(f.name.encode())
        h.update(content)
    return h.hexdigest(), per_file
