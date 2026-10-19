"""Simulation performance metrics with Monte Carlo standard errors (Morris, White & Crowther, Stat Med 2019).

Definitions are fixed in docs/estimands.md before any main run. Failure handling: a replicate with no estimate or no finite
SE is a *failure*. Bias, empirical SE, model-SE ratio, MSE and CI width use successful replicates; coverage and rejection
count failures as non-coverage / non-rejection and are also reported among successes only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bias(est, truth):
    est = np.asarray(est, float)
    return float(est.mean() - truth), float(est.std(ddof=1) / np.sqrt(len(est)))


def emp_se(est):
    est = np.asarray(est, float)
    sd = float(est.std(ddof=1))
    return sd, sd / np.sqrt(2 * (len(est) - 1))


def model_se_ratio(se, est):
    se, est = np.asarray(se, float), np.asarray(est, float)
    R = len(est)
    mod = float(np.sqrt(np.mean(se ** 2)))
    emp, emp_mc = emp_se(est)
    mod_mc = float(np.sqrt(np.var(se ** 2, ddof=1) / (4 * R * mod ** 2)))
    ratio = mod / emp
    return ratio, float(ratio * np.sqrt((mod_mc / mod) ** 2 + (emp_mc / emp) ** 2))


def proportion(x):
    x = np.asarray(x, float)
    p = float(x.mean())
    return p, float(np.sqrt(p * (1 - p) / len(x)))


def mse(est, truth):
    e2 = (np.asarray(est, float) - truth) ** 2
    R = len(e2)
    m = float(e2.mean())
    return m, float(np.sqrt(np.sum((e2 - m) ** 2) / (R * (R - 1))))


# estimand each method targets (a method is scored only against this truth)
TARGET = {
    "cox_multivariable": "conditional_log_hr",
    "cox_iptw": "marginal_log_hr",
    "cox_unadjusted": "marginal_log_hr",
    "rmst24_km": "rmst_diff_24", "rmst24_iptw": "rmst_diff_24", "rmst24_gcomp": "rmst_diff_24", "rmst24_aipw": "rmst_diff_24",
    "rmst12_km": "rmst_diff_12", "rmst12_iptw": "rmst_diff_12", "rmst12_gcomp": "rmst_diff_12", "rmst12_aipw": "rmst_diff_12",
    "surv12_km": "surv_diff_12", "surv24_km": "surv_diff_24",
}
TRUTH_MCSE = {"conditional_log_hr": None, "marginal_log_hr": "marginal_log_hr_se_approx"}


def summarise(rows: pd.DataFrame, truth: dict) -> pd.DataFrame:
    """rows: one line per replicate x method with columns method, est, se, lo, hi, failure ('' if none)."""
    out = []
    for method, g in rows.groupby("method"):
        key = TARGET[method]
        th = truth[key]
        th_mcse = truth.get(TRUTH_MCSE.get(key) or key + "_mcse", 0.0) or 0.0
        R = len(g)
        ok = g[g["failure"].fillna("") == ""]
        n_ok = len(ok)
        rec = {"method": method, "estimand": key, "truth": th, "truth_mcse": th_mcse, "R": R, "n_fail": R - n_ok,
               "fail_rate": (R - n_ok) / R, "fail_rate_mcse": proportion(np.r_[np.ones(R - n_ok), np.zeros(n_ok)])[1]}
        if n_ok >= 5:
            b, b_mc = bias(ok["est"], th)
            es, es_mc = emp_se(ok["est"])
            ratio, ratio_mc = model_se_ratio(ok["se"], ok["est"])
            m, m_mc = mse(ok["est"], th)
            covered = ((ok["lo"] <= th) & (th <= ok["hi"])).astype(float)
            rej_ok = ((ok["lo"] > 0) | (ok["hi"] < 0)).astype(float)
            cov_all = np.r_[covered.values, np.zeros(R - n_ok)]
            rej_all = np.r_[rej_ok.values, np.zeros(R - n_ok)]
            rec.update({
                "bias": b, "bias_mcse": b_mc, "emp_se": es, "emp_se_mcse": es_mc,
                "model_se_ratio": ratio, "model_se_ratio_mcse": ratio_mc, "mse": m, "mse_mcse": m_mc,
                "coverage": proportion(cov_all)[0], "coverage_mcse": proportion(cov_all)[1],
                "coverage_among_ok": proportion(covered)[0],
                "rejection": proportion(rej_all)[0], "rejection_mcse": proportion(rej_all)[1],
                "median_ci_width": float((ok["hi"] - ok["lo"]).median()),
                "ci_width_iqr": float((ok["hi"] - ok["lo"]).quantile(0.75) - (ok["hi"] - ok["lo"]).quantile(0.25)),
                "truth_precision_ok": bool(th_mcse < 0.1 * es),
                "flag_gt5pct_failures": bool((R - n_ok) / R > 0.05),
            })
        out.append(rec)
    return pd.DataFrame(out)


def atrisk_summary(diag: pd.DataFrame) -> pd.DataFrame:
    """Median and minimum number at risk at 12 and 24 months per arm, and the number of replicates below 20."""
    rows = []
    for col in ["risk24_io", "risk24_chemo", "risk12_io", "risk12_chemo"]:
        v = diag[col]
        rows.append({"quantity": col, "median": float(v.median()), "min": int(v.min()), "p10": float(np.percentile(v, 10)),
                     "replicates_below_20": int((v < 20).sum()), "R": len(v)})
    return pd.DataFrame(rows)
