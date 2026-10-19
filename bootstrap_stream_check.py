"""Check whether seeding the bootstrap with the data seed matters.

For the first N main seeds of a cell the cohort is regenerated and the four bootstrap-based estimators (IPTW-KM and g-computation
RMST at 24 and 12 months) are run twice: with the ORIGINAL stream default_rng(seed) (what replicate() uses) and with an
INDEPENDENT stream default_rng([seed, 1]). Decision rule (fixed in docs/estimands.md section 9 before running):
  (i) point estimates identical; (ii) mean log ratio of SE (independent / original) within 2 Monte Carlo SE of 0;
  (iii) paired difference in 95% coverage within 2 Monte Carlo SE of 0.

    python bootstrap_stream_check.py run --cell S0 --N 100 --workers 5
    python bootstrap_stream_check.py report --cell S0
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scenario_study as S  # noqa: E402
from oncrwd import analysis as A  # noqa: E402
from oncrwd import rmst  # noqa: E402

B = 100


def one(args):
    cell, seed = args
    c, _ = S.make_cohort(S.C.N_PATIENTS, seed, S.cell_opts(cell))
    X = A.make_X(c, 10)
    covs = A.varying(X)
    d = pd.concat([c[["os_months", "event", "io"]], X[covs]], axis=1)
    rows = []
    for stream, rng in (("original", np.random.default_rng(seed)), ("independent", np.random.default_rng([seed, 1]))):
        for tau, tag in ((24.0, "rmst24"), (12.0, "rmst12")):
            for name, fn in (("iptw", lambda r: rmst.rmst_iptw(d, X, covs, tau, B, r)), ("gcomp", lambda r: rmst.rmst_gcomp(d, X, covs, tau, B, r))):
                r = fn(rng)
                rows.append({"cell": cell, "seed": seed, "stream": stream, "method": f"{tag}_{name}", "est": r.get("estimate", np.nan),
                             "se": r.get("se", np.nan), "lo": r.get("lo", np.nan), "hi": r.get("hi", np.nan), "failure": r.get("failure", "")})
    return rows


def run(cell, N, workers):
    out = S.OUT / f"bootstrap_check_{cell}.csv"
    done = set(pd.read_csv(out)["seed"]) if out.exists() else set()
    todo = [(cell, s) for s in S.seeds_for(cell, N) if s not in done]
    with Pool(workers) as p:
        for k, rows in enumerate(p.imap_unordered(one, todo), 1):
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
            if k % 10 == 0:
                print(f"{cell}: {k}/{len(todo)}", flush=True)


def load_truth_any(cell):
    try:
        return S.load_truth(cell)
    except SystemExit:
        return json.loads((S.OUT / f"truth_{cell}.json").read_text())      # earlier unhashed file (same seeds, same settings)


def report(cell):
    df = pd.read_csv(S.OUT / f"bootstrap_check_{cell}.csv")
    th = load_truth_any(cell)
    key = {"rmst24": th["rmst_diff_24"], "rmst12": th["rmst_diff_12"]}
    out = []
    for m, g in df.groupby("method"):
        o = g[g.stream == "original"].set_index("seed").sort_index()
        i = g[g.stream == "independent"].set_index("seed").sort_index()
        ok = (o.failure.fillna("") == "") & (i.failure.fillna("") == "")
        o, i = o[ok], i[ok]
        truth = key[m.split("_")[0]]
        lr = np.log(i.se / o.se)
        cov_o = ((o.lo <= truth) & (truth <= o.hi)).astype(float)
        cov_i = ((i.lo <= truth) & (truth <= i.hi)).astype(float)
        dcov = cov_i - cov_o
        n = len(lr)
        rec = {"cell": cell, "method": m, "n_pairs": n, "max_abs_est_diff": float((o.est - i.est).abs().max()),
               "mean_log_se_ratio": float(lr.mean()), "mcse_log_se_ratio": float(lr.std(ddof=1) / np.sqrt(n)),
               "coverage_original": float(cov_o.mean()), "coverage_independent": float(cov_i.mean()),
               "coverage_diff": float(dcov.mean()), "mcse_coverage_diff": float(dcov.std(ddof=1) / np.sqrt(n))}
        rec["rule_i"] = rec["max_abs_est_diff"] < 1e-12
        rec["rule_ii"] = abs(rec["mean_log_se_ratio"]) < 2 * rec["mcse_log_se_ratio"]
        rec["rule_iii"] = abs(rec["coverage_diff"]) < 2 * rec["mcse_coverage_diff"] if rec["mcse_coverage_diff"] > 0 else rec["coverage_diff"] == 0
        out.append(rec)
    res = pd.DataFrame(out)
    res.to_csv(S.OUT / f"bootstrap_check_{cell}_report.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(res.round(4).to_string(index=False))
    print("ALL RULES MET" if res[["rule_i", "rule_ii", "rule_iii"]].all().all() else "RULE NOT MET")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    run(a.cell, a.N, a.workers) if a.cmd == "run" else report(a.cell)
