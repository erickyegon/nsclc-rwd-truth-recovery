"""Monte-Carlo validation: regenerate the EHR under many seeds, rerun the whole pipeline,
and quantify bias / coverage of each estimator against the known simulated truth.

    python simulation_study.py --seeds 10
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from oncrwd import analysis as A  # noqa: E402
from oncrwd import cohort as K  # noqa: E402
from oncrwd import config as C  # noqa: E402
from oncrwd import generate, lot  # noqa: E402

warnings.filterwarnings("ignore")


def one_run(seed: int, n: int) -> list[dict]:
    tmp = Path(tempfile.mkdtemp(prefix="oncsim_"))
    try:
        generate.generate(n=n, seed=seed, raw_dir=tmp / "raw", truth_dir=tmp / "truth")
        clean, _ = K.run_qc(K.load_raw(tmp / "raw"))
        lines = lot.derive_lines(lot.build_admin_table(clean["medications"]))
        coh, _ = K.build_cohort(clean, lines)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    era = pd.Timestamp(C.IO_ERA_START)
    c = coh[(coh["index_date"] >= era) & (coh["egfr"] != "Positive") & coh["regimen_class"].isin(["IO + chemo", "Platinum chemo"])].copy()
    c["io"] = (c["regimen_class"] == "IO + chemo").astype(int)
    X = A.make_X(c, 10)
    covs = A.varying(X)
    d = pd.concat([c[["os_months", "event", "io"]], X[covs]], axis=1)
    ps, w = A.propensity_weights(c, "io", A.varying(A.make_X(c)))
    d["w"] = w
    out = []
    for name, m in [("Unadjusted", A.cox_fit(d, ["io"])), ("Multivariable Cox", A.cox_fit(d, ["io"] + covs)),
                    ("IPTW", A.cox_fit(d, ["io"], weights="w"))]:
        s = m.summary.loc["io"]
        out.append({"seed": seed, "estimand": "IO+chemo vs chemo", "method": name, "log_hr": s["coef"],
                    "lo": np.log(s["exp(coef) lower 95%"]), "hi": np.log(s["exp(coef) upper 95%"]),
                    "truth": C.TRUE_LOG_HR["io_chemo_vs_chemo"]})
    itb = A.immortal_time_demo(coh)
    for _, r in itb.iterrows():
        out.append({"seed": seed, "estimand": "2L therapy (null)", "method": r["method"], "log_hr": np.log(r["hr"]),
                    "lo": np.log(r["hr_low"]), "hi": np.log(r["hr_high"]), "truth": 0.0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=C.N_PATIENTS)
    a = ap.parse_args()
    rows = []
    for s in range(1, a.seeds + 1):
        rows += one_run(1000 + s, a.n)
        print(f"seed {s}/{a.seeds} done", flush=True)
    df = pd.DataFrame(rows)
    df["covered"] = (df["lo"] <= df["truth"]) & (df["truth"] <= df["hi"])
    summ = (df.groupby(["estimand", "method"])
              .agg(mean_log_hr=("log_hr", "mean"), truth=("truth", "first"), empirical_sd=("log_hr", "std"),
                   coverage=("covered", "mean"), runs=("seed", "nunique")))
    summ["bias"] = summ["mean_log_hr"] - summ["truth"]
    C.RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(C.RESULTS_DIR / "simulation_study_runs.csv", index=False)
    summ.round(3).to_csv(C.RESULTS_DIR / "simulation_study_summary.csv")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
