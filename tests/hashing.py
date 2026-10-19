"""Strict content hash of generator output for same-machine comparisons (line endings normalised to LF)."""
from __future__ import annotations

import hashlib
from pathlib import Path

CRLF, LF = bytes([13, 10]), bytes([10])


def strict_hash(raw_dir: Path, truth_dir: Path):
    """SHA-256 over file names and LF-normalised bytes of the six raw tables and the two truth tables, plus per-file digests."""
    h = hashlib.sha256()
    per_file = {}
    for f in sorted(Path(raw_dir).glob("*.csv")) + [Path(truth_dir) / "truth_patients.csv", Path(truth_dir) / "truth_lines.csv"]:
        content = f.read_bytes().replace(CRLF, LF)
        per_file[f.name] = hashlib.sha256(content).hexdigest()[:12]
        h.update(f.name.encode())
        h.update(content)
    return h.hexdigest(), per_file
