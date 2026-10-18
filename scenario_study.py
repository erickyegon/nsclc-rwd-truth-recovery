"""Phase 1 scenario study: truth, replicates and summaries for the IO+chemo vs chemo contrast.

    python scenario_study.py truth     --cell S0                      # large counterfactual truth for one cell
    python scenario_study.py run       --cell S0 --R 1000 --workers 10  # replicates (resumable)
    python scenario_study.py summarise --cell S0                      # metrics with Monte Carlo SE
    python scenario_study.py calibrate                                # confounding / positivity levels (pilot seeds)

Seed policy: pilot seeds 7_000_000+, truth populations 9_000_000+, main replicates 1_001 + 10_000 * cell_index + r.
Earlier exploratory seeds (101, 102, 201-220, 301-308) are burned and never reused.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from oncrwd import analysis as A  # noqa: E402
from oncrwd import cohort as K  # noqa: E402
from oncrwd import config as C  # noqa: E402
from oncrwd import generate, lot, metrics, rmst, truth  # noqa: E402

OUT = ROOT / "results" / "phase1"
CAL_FILE = OUT / "calibration.json"
TRUE_EFFECT = C.TRUE_LOG_HR["io_chemo_vs_chemo"]

# cell id -> (scenario label, replicates, generator options). Calibrated levels are read from calibration.json.
CELL_ORDER = ["S0", "S3_null", "S3_m070", "S3_p020", "S1_mod", "S1_strong", "S2_mod", "S2_severe"]
CELL_R = {"S0": 1000}


def cell_opts(cell: str) -> dict:
    cal = json.loads(CAL_FILE.read_text()) if CAL_FILE.exists() else {}
    table = {
        "S0": {},
        "S3_null": {"io_chemo_effect": 0.0},
        "S3_m070": {"io_chemo_effect": -0.70},
        "S3_p020": {"io_chemo_effect": 0.20},
        "S1_mod": {"confounding_gamma": cal.get("gamma_mod")},
        "S1_strong": {"confounding_gamma": cal.get("gamma_strong")},
        "S2_mod": {"positivity_mult": cal.get("posmult_mod")},
        "S2_severe": {"positivity_mult": cal.get("posmult_severe")},
    }
    o = table[cell]
    if any(v is None for v in o.values()):
        raise SystemExit(f"cell {cell} needs calibration.json (run `calibrate` first)")
    return o


def seeds_for(cell: str, R: int) -> list[int]:
    return [1001 + 10_000 * CELL_ORDER.index(cell) + r for r in range(R)]


# ----------------------------------------------------------------------------- shared pipeline helpers
def make_cohort(n: int, seed: int, opts: dict, keep_truth: bool = False):
    tmp = Path(tempfile.mkdtemp(prefix="p1_"))
    try:
        generate.generate(n=n, seed=seed, raw_dir=tmp / "raw", truth_dir=tmp / "truth", opts=opts)
        clean, _ = K.run_qc(K.load_raw(tmp / "raw"))
        lines = lot.derive_lines(lot.build_admin_table(clean["medications"]))
        coh, _ = K.build_cohort(clean, lines)
        tp = pd.read_csv(tmp / "truth" / "truth_patients.csv") if keep_truth else None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    era = pd.Timestamp(C.IO_ERA_START)
    c = coh[(coh["index_date"] >= era) & (coh["egfr"] != "Positive") & coh["regimen_class"].isin(["IO + chemo", "Platinum chemo"])].copy()
    c["io"] = (c["regimen_class"] == "IO + chemo").astype(int)
    return c, tp


# ----------------------------------------------------------------------------- one replicate
def replicate(args):
    cell, seed, n, B = args
    t0 = time.time()
    rng = np.random.default_rng(seed)
    opts = cell_opts(cell)
    rows, diag = [], {"cell": cell, "seed": seed}
    try:
        c, _ = make_cohort(n, seed, opts)
        X = A.make_X(c, 10)
        covs = A.varying(X)
        d = pd.concat([c[["os_months", "event", "io"]], X[covs]], axis=1)
        ps, w = A.propensity_weights(c, "io", A.varying(A.make_X(c)))
        d["w"] = w
        diag.update({"n_cohort": len(c), "n_io": int(c["io"].sum()), "n_chemo": int((1 - c["io"]).sum()),
                     "ps_min": float(ps.min()), "ps_share_lt05": float(((ps < 0.05) | (ps > 0.95)).mean())})
        for tau in (24, 12):
            for arm, g in (("io", 1), ("chemo", 0)):
                diag[f"risk{tau}_{arm}"] = int(((d["io"] == g) & (d["os_months"] >= tau)).sum())
    except Exception as e:  # noqa: BLE001
        return {"cell": cell, "seed": seed, "fatal": str(e)}, []

    def add(method, res):
        rows.append({"cell": cell, "seed": seed, "method": method, "est": res.get("estimate", np.nan), "se": res.get("se", np.nan),
                     "lo": res.get("lo", np.nan), "hi": res.get("hi", np.nan), "failure": res.get("failure", "")})

    def cox(cols, weights=None):
        try:
            s = A.cox_fit(d, cols, weights=weights).summary.loc["io"]
            return {"estimate": s["coef"], "se": s["se(coef)"], "lo": np.log(s["exp(coef) lower 95%"]), "hi": np.log(s["exp(coef) upper 95%"])}
        except Exception as e:  # noqa: BLE001
            return {"failure": f"cox: {e}"}

    add("cox_unadjusted", cox(["io"]))
    add("cox_multivariable", cox(["io"] + covs))
    add("cox_iptw", cox(["io"], weights="w"))
    for tau, tag in ((24.0, "rmst24"), (12.0, "rmst12")):
        add(f"{tag}_km", rmst.rmst_km(d, tau))
        add(f"{tag}_iptw", rmst.rmst_iptw(d, X, covs, tau, B, rng))
        add(f"{tag}_gcomp", rmst.rmst_gcomp(d, X, covs, tau, B, rng))
        add(f"{tag}_aipw", rmst.rmst_aipw(d, X, covs, tau))
    add("surv12_km", rmst.surv_diff_km(d, 12.0))
    add("surv24_km", rmst.surv_diff_km(d, 24.0))
    diag["secs"] = round(time.time() - t0, 1)
    return diag, rows


def run(cell, R, workers, n, B):
    OUT.mkdir(parents=True, exist_ok=True)
    rf, df_ = OUT / f"rows_{cell}.csv", OUT / f"diag_{cell}.csv"
    done = set(pd.read_csv(df_)["seed"]) if df_.exists() else set()
    todo = [(cell, s, n, B) for s in seeds_for(cell, R) if s not in done]
    print(f"{cell}: {len(done)} done, {len(todo)} to run", flush=True)
    with Pool(workers) as p:
        for k, (diag, rows) in enumerate(p.imap_unordered(replicate, todo), 1):
            if "fatal" in diag:
                print("FATAL", diag, flush=True)
                continue
            pd.DataFrame(rows).to_csv(rf, mode="a", header=not rf.exists(), index=False)
            pd.DataFrame([diag]).to_csv(df_, mode="a", header=not df_.exists(), index=False)
            if k % 25 == 0:
                print(f"  {cell}: {k}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- truth
def _truth_chunk(args):
    seed, opts = args
    c, tp = make_cohort(6000, seed, opts, keep_truth=True)
    j = tp.set_index("patient_id").loc[c.index.intersection(tp["patient_id"])]
    return j[["age", "male", "hist", "stage", "ecog", "n_comorb", "brain", "egfr"]]


def build_truth(cell, chunks, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    opts = cell_opts(cell)
    seeds = [(9_000_000 + 1000 * CELL_ORDER.index(cell) + k, opts) for k in range(chunks)]
    with Pool(workers) as p:
        parts = p.map(_truth_chunk, seeds)
    cov = pd.concat(parts, ignore_index=True)
    effect = opts.get("io_chemo_effect", TRUE_EFFECT)
    res = truth.counterfactual_truth(cov, effect, m=20, seed=2026_0001 + CELL_ORDER.index(cell))
    for tau in (24, 12):
        res[f"rmst_diff_{tau}_analytic"] = truth.analytic_rmst_diff(cov, effect, float(tau))
    res["opts"] = opts
    (OUT / f"truth_{cell}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in res if "rmst" in k or "n_pop" in k or "marginal" in k}, indent=1))


def summarise(cell):
    rows = pd.read_csv(OUT / f"rows_{cell}.csv")
    diag = pd.read_csv(OUT / f"diag_{cell}.csv")
    th = json.loads((OUT / f"truth_{cell}.json").read_text())
    s = metrics.summarise(rows, th)
    s.insert(0, "cell", cell)
    s.to_csv(OUT / f"summary_{cell}.csv", index=False)
    a = metrics.atrisk_summary(diag)
    a.insert(0, "cell", cell)
    a.to_csv(OUT / f"atrisk_{cell}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(s[["method", "R", "n_fail", "truth", "bias", "bias_mcse", "emp_se", "model_se_ratio", "coverage", "coverage_mcse", "rejection", "mse"]].round(4).to_string(index=False))
    print(a.to_string(index=False))


# ----------------------------------------------------------------------------- calibration
def _pilot_chunk(args):
    seed, opts = args
    c, _ = make_cohort(6000, seed, opts)
    X = A.make_X(c, 10)
    covs = A.varying(X)
    ps, _ = A.propensity_weights(c, "io", A.varying(A.make_X(c)))
    c = c.assign(ps=ps)
    return c[["os_months", "event", "io", "ps"]]


def calibrate(workers, chunks=40):
    OUT.mkdir(parents=True, exist_ok=True)
    from lifelines import CoxPHFitter
    out = {}
    # (a) LP standardisation from the baseline target population (truth populations carry the true covariates)
    with Pool(workers) as p:
        parts = p.map(_truth_chunk, [(7_100_000 + k, {}) for k in range(chunks)])
    lp = truth.linear_predictor(pd.concat(parts, ignore_index=True))
    out["LP_MEAN"], out["LP_SD"] = float(lp.mean()), float(lp.std())
    print("LP mean/sd", out["LP_MEAN"], out["LP_SD"], flush=True)
    lpo = {"lp_mean": out["LP_MEAN"], "lp_sd": out["LP_SD"]}
    # (b) unadjusted-bias grid for confounding gamma (pooled pilot cohorts)
    grid = {}
    for g in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        with Pool(workers) as p:
            pooled = pd.concat(p.map(_pilot_chunk, [(7_200_000 + int(g * 100) * 100 + k, {"confounding_gamma": g, **lpo}) for k in range(chunks)]))
        est = float(CoxPHFitter().fit(pooled[["os_months", "event", "io"]], "os_months", "event").params_["io"])
        grid[g] = est
        print(f"gamma={g}: unadjusted log-HR {est:.3f}", flush=True)
    base = grid[0.0]
    gs = np.array(sorted(grid))
    bias = np.array([grid[g] - base for g in gs])        # negative: confounding makes IO look better
    for name, target in (("gamma_mod", -0.10), ("gamma_strong", -0.25)):
        out[name] = float(np.interp(target, bias[::-1], gs[::-1]))
    out["gamma_grid_unadjusted_loghr"] = grid
    # (c) positivity grid: min propensity and share of patients with PS outside [0.05, 0.95]
    pgrid = {}
    for m in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        with Pool(workers) as p:
            pooled = pd.concat(p.map(_pilot_chunk, [(7_300_000 + int(m * 10) * 100 + k, {"positivity_mult": m}) for k in range(chunks // 2)]))
        pgrid[m] = {"ps_min": float(pooled["ps"].min()), "ps_p01": float(pooled["ps"].quantile(0.01)),
                    "share_outside_05_95": float(((pooled["ps"] < 0.05) | (pooled["ps"] > 0.95)).mean())}
        print(f"posmult={m}: {pgrid[m]}", flush=True)
    out["positivity_grid"] = pgrid
    # moderate: smallest multiplier with ps_p01 < 0.10; severe: smallest with >= 10% outside [0.05, 0.95]
    mods = [m for m in sorted(pgrid) if pgrid[m]["ps_p01"] < 0.10]
    sev = [m for m in sorted(pgrid) if pgrid[m]["share_outside_05_95"] >= 0.10]
    out["posmult_mod"] = float(mods[0]) if mods else 3.0
    out["posmult_severe"] = float(sev[0]) if sev else 6.0
    CAL_FILE.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if not isinstance(out[k], dict)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["truth", "run", "summarise", "calibrate"])
    ap.add_argument("--cell")
    ap.add_argument("--R", type=int)
    ap.add_argument("--n", type=int, default=C.N_PATIENTS)
    ap.add_argument("--B", type=int, default=100, help="bootstrap resamples for IPTW-RMST and g-computation")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=40, help="truth/calibration population chunks of 6,000 patients")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a.workers, a.chunks)
    elif a.cmd == "truth":
        build_truth(a.cell, a.chunks, a.workers)
    elif a.cmd == "run":
        run(a.cell, a.R or CELL_R.get(a.cell, 500), a.workers, a.n, a.B)
    else:
        summarise(a.cell)
