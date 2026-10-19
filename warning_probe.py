"""Probe: which warnings (in particular convergence warnings) do real study replicates emit?

scenario_study silences all warnings at import. This probe runs the real replicate() inside warnings.catch_warnings with every
warning switched back on, for a handful of seeds, and tallies category and message. Results are descriptive only.

    python warning_probe.py --cell S0 --N 20 --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import scenario_study as S  # noqa: E402


def one(args):
    cell, seed = args
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diag, rows = S.replicate((cell, seed, 6000, 100))
    c = Counter()
    for w in caught:
        c[(w.category.__name__, str(w.message).strip().splitlines()[0][:110])] += 1
    failures = [r["failure"] for r in rows if r.get("failure")]
    return seed, c, failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    seeds = S.seeds_for(a.cell, a.N)
    tot, per_seed, fails = Counter(), {}, []
    with Pool(a.workers) as p:
        for seed, c, f in p.imap_unordered(one, [(a.cell, s) for s in seeds]):
            tot.update(c)
            per_seed[seed] = sum(c.values())
            fails += f
    out = {"cell": a.cell, "n_replicates": a.N, "totals": {f"{k[0]}: {k[1]}": v for k, v in tot.most_common()},
           "replicates_with_any_warning": sum(1 for v in per_seed.values() if v), "method_failures": fails}
    (ROOT.parent / f"warning_probe_{a.cell}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
