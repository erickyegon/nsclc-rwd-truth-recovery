"""Cross-platform reproducibility check for the Phase 1 replicates.

Regenerates the first N main replicates of a cell on the machine running this script (the real replicate(): generator, cohort,
estimators, bootstrap) and compares the estimates, standard errors, interval ends and coverage indicators with the rows produced
on Windows during the study (results/phase1/crossplatform_reference_windows.csv, extracted from rows_<cell>.csv).

TOLERANCE, fixed before the first comparison (see docs/estimands.md section 9):
  * estimate, SE, interval lower and upper end: |a - b| <= 1e-6 * max(|a|, |b|, 1e-3)   (relative, with an absolute floor)
  * 95% coverage indicator (lo <= truth <= hi): no mismatches
  * failure text: identical;  cohort and at-risk counts: identical;  ps_min: same relative tolerance as estimates

    python crossplatform_check.py run     --cell S0 --N 20 --workers 2 --out new_S0.csv
    python crossplatform_check.py compare --cell S0 --new new_S0.csv [--summary summary.md]
    python crossplatform_check.py extract --cells S0 S3_null --N 20       # build the Windows reference from the study rows
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import scenario_study as S  # noqa: E402

REL_TOL = 1e-6
ABS_FLOOR = 1e-3
REF = S.OUT / "crossplatform_reference_windows.csv"
REF_DIAG = S.OUT / "crossplatform_reference_windows_diag.csv"
VALUE_COLS = ["est", "se", "lo", "hi"]
DIAG_EXACT = ["n_cohort", "n_io", "n_chemo", "risk24_io", "risk24_chemo", "risk12_io", "risk12_chemo"]


def run_one(args):
    cell, seed = args
    diag, rows = S.replicate((cell, seed, 6000, 100))
    return diag, rows


def run(cell, N, workers, out):
    seeds = S.seeds_for(cell, N)
    with Pool(workers) as p:
        res = p.map(run_one, [(cell, s) for s in seeds])
    rows = [r for _, rs in res for r in rs]
    diags = [d for d, _ in res]
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(diags).to_csv(Path(out).with_suffix(".diag.csv"), index=False)
    print(f"wrote {len(rows)} rows for {len(seeds)} seeds -> {out}")


def extract(cells, N):
    rows, diags = [], []
    for c in cells:
        seeds = set(S.seeds_for(c, N))
        r = pd.read_csv(S.OUT / f"rows_{c}.csv")
        d = pd.read_csv(S.OUT / f"diag_{c}.csv")
        rows.append(r[r.seed.isin(seeds)])
        diags.append(d[d.seed.isin(seeds)])
    pd.concat(rows).sort_values(["cell", "seed", "method"]).to_csv(REF, index=False)
    pd.concat(diags).sort_values(["cell", "seed"]).to_csv(REF_DIAG, index=False)
    print("reference rows:", sum(len(x) for x in rows), "diag rows:", sum(len(x) for x in diags))


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), ABS_FLOOR)


def compare(cell, new_path, summary):
    ref = pd.read_csv(REF)
    ref = ref[ref.cell == cell]
    new = pd.read_csv(new_path)
    new = new[new.cell == cell]
    th = S.load_truth(cell)
    key = {"rmst24": th["rmst_diff_24"], "rmst12": th["rmst_diff_12"]}
    from oncrwd import metrics as M
    seeds_new = set(new.seed)
    ref = ref[ref.seed.isin(seeds_new)]
    m = ref.merge(new, on=["cell", "seed", "method"], suffixes=("_win", "_new"), how="outer", indicator=True)
    missing = int((m["_merge"] != "both").sum())
    m = m[m["_merge"] == "both"].copy()
    lines = [f"### Cross-platform comparison, cell {cell}: {len(seeds_new)} seeds, {len(m)} method rows (Windows study rows vs this machine)", "",
             f"Tolerance fixed in advance: relative {REL_TOL:g} with absolute floor {ABS_FLOOR:g} for est, se, lo, hi; coverage indicators and counts identical.", ""]
    ok = missing == 0
    lines.append(f"- rows present in only one file: **{missing}**")
    lines += ["", "| field | max abs diff | max rel diff | rows over tolerance |", "|---|---|---|---|"]
    for c in VALUE_COLS:
        a, b = m[f"{c}_win"], m[f"{c}_new"]
        both_nan = a.isna() & b.isna()
        d = (a - b).abs().where(~both_nan, 0.0)
        r = pd.Series(rel(a.fillna(0), b.fillna(0)), index=m.index).where(~both_nan, 0.0)
        over = int((r > REL_TOL).sum() + (a.isna() ^ b.isna()).sum())
        ok &= over == 0
        lines.append(f"| {c} | {d.max():.3g} | {r.max():.3g} | {over} |")
    fw, fn = m["failure_win"].fillna(""), m["failure_new"].fillna("")
    fail_mismatch = int((fw != fn).sum())
    ok &= fail_mismatch == 0

    def cov(df, sfx):
        t = df["method"].map(lambda x: M.TARGET[x]).map(lambda k: th[k])
        return ((df[f"lo_{sfx}"] <= t) & (t <= df[f"hi_{sfx}"])).astype(int)
    cm = int((cov(m, "win") != cov(m, "new")).sum())
    ok &= cm == 0
    lines += ["", f"- coverage-indicator mismatches: **{cm}** of {len(m)}", f"- failure-text mismatches: **{fail_mismatch}**"]
    # diagnostics
    dref = pd.read_csv(REF_DIAG)
    dref = dref[(dref.cell == cell) & dref.seed.isin(seeds_new)]
    dnew = pd.read_csv(Path(new_path).with_suffix(".diag.csv"))
    dm = dref.merge(dnew, on=["cell", "seed"], suffixes=("_win", "_new"))
    dmis = {c: int((dm[f"{c}_win"] != dm[f"{c}_new"]).sum()) for c in DIAG_EXACT}
    ps_rel = float(rel(dm["ps_min_win"], dm["ps_min_new"]).max())
    ok &= sum(dmis.values()) == 0 and ps_rel <= REL_TOL
    lines += [f"- cohort and at-risk count mismatches: {dmis}", f"- max relative difference in minimum propensity score: {ps_rel:.3g}",
              "", f"**Result: {'WITHIN TOLERANCE' if ok else 'OUTSIDE TOLERANCE'}**"]
    text = "\n".join(lines)
    print(text)
    if summary:
        with open(summary, "a", encoding="utf8") as f:
            f.write(text + "\n\n")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "compare", "extract"])
    ap.add_argument("--cell")
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="crossplatform_new.csv")
    ap.add_argument("--new")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.cell, a.N, a.workers, a.out)
    elif a.cmd == "extract":
        extract(a.cells, a.N)
    else:
        sys.exit(0 if compare(a.cell, a.new, a.summary) else 1)
