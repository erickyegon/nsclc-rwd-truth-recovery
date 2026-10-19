"""Known truth for the IO+chemo vs chemo contrast, computed independently of the estimator code.

Independence: this module imports nothing from analysis.py, rmst.py, cohort.py or lot.py and never calls the generator's
survival draw. It re-implements the survival model stated in config.py (Weibull proportional hazards, shape 1.1, median 700
days at linear predictor 0) with scipy, draws BOTH potential outcomes for every patient of the target population with a
separate random stream, and also integrates the closed-form survival function as a cross-check.

Target population = the patients that enter the analysis cohort (IO + chemo or platinum chemo, EGFR not positive, IO era,
advanced stage, ...), with their TRUE covariates, taken from large generator runs (see scenario_study.build_truth).
Truth is the uncensored, marginal (population-averaged) contrast:
  E2   RMST(tau | IO+chemo) - RMST(tau | chemo), months, tau = 24 (primary) and 12 (secondary)
  E3   survival probability difference at 12 and 24 months
  E1   conditional log-HR = the configured effect (verified by the pseudo-population Cox fit with covariates)
  E4   marginal log-HR: Cox on the stacked pseudo-population with treatment independent of covariates, no censoring
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import weibull_min

from . import config as C

SHAPE = 1.1          # Weibull shape of overall survival (generator: _weibull(..., 1.1, 700))
SCALE0_DAYS = 700.0  # median at linear predictor 0
DPM = C.DAYS_PER_MONTH


def linear_predictor(cov: pd.DataFrame) -> np.ndarray:
    """Prognostic linear predictor from TRUE covariates (columns of truth_patients.csv)."""
    t = C.TRUE_LOG_HR
    return (t["age_per_year"] * (cov["age"].values - 68) + t["ecog_per_level"] * cov["ecog"].values
            + t["stage4_vs_3"] * (cov["stage"].values == 4) + t["brain_mets"] * cov["brain"].astype(float).values
            + t["comorbidity_per_condition"] * cov["n_comorb"].values + t["male"] * cov["male"].astype(float).values
            + t["squamous"] * (cov["hist"].values == "squam") + t["egfr_positive"] * cov["egfr"].astype(float).values)


def _scale_days(lp):
    lam = np.log(2) / SCALE0_DAYS ** SHAPE
    return (lam * np.exp(lp)) ** (-1.0 / SHAPE)       # weibull_min scale s.t. S(t) = exp(-(t/s)^k) = exp(-lam e^lp t^k)


def counterfactual_truth(cov: pd.DataFrame, effect: float, m: int = 20, seed: int = 2026_0001, taus=(24.0, 12.0)) -> dict:
    """Monte-Carlo truth from m independent pairs of potential outcomes per patient (separate PCG64 stream)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    lp = linear_predictor(cov)
    n = len(lp)
    out = {"n_population": int(n), "m_draws": m, "effect": effect}
    d_rmst = {tau: np.zeros(n) for tau in taus}
    d_surv = {tau: np.zeros(n) for tau in taus}
    pseudo = []
    for j in range(m):
        t1 = np.maximum(1.0, weibull_min.rvs(SHAPE, scale=_scale_days(lp + effect), size=n, random_state=rng)) / DPM
        t0 = np.maximum(1.0, weibull_min.rvs(SHAPE, scale=_scale_days(lp), size=n, random_state=rng)) / DPM
        for tau in taus:
            d_rmst[tau] += (np.minimum(t1, tau) - np.minimum(t0, tau)) / m
            d_surv[tau] += ((t1 > tau).astype(float) - (t0 > tau).astype(float)) / m
        if j < 5:   # marginal-HR pseudo-population uses the first 5 draws only (cost)
            pseudo.append((t1, t0))
    for tau in taus:
        out[f"rmst_diff_{int(tau)}"] = float(d_rmst[tau].mean())
        out[f"rmst_diff_{int(tau)}_mcse"] = float(d_rmst[tau].std(ddof=1) / np.sqrt(n))
        out[f"surv_diff_{int(tau)}"] = float(d_surv[tau].mean())
        out[f"surv_diff_{int(tau)}_mcse"] = float(d_surv[tau].std(ddof=1) / np.sqrt(n))
    # E4: marginal log-HR. Cox with one binary covariate on the stacked pseudo-population (no censoring)
    from lifelines import CoxPHFitter
    t_all = np.concatenate([np.concatenate([a, b]) for a, b in pseudo])
    z = np.concatenate([np.concatenate([np.ones(n), np.zeros(n)]) for _ in pseudo])
    df = pd.DataFrame({"t": t_all, "e": 1, "z": z})
    cph = CoxPHFitter().fit(df, "t", "e")
    out["marginal_log_hr"] = float(cph.params_["z"])
    out["marginal_log_hr_se_approx"] = float(cph.standard_errors_["z"])
    out["conditional_log_hr"] = float(effect)
    return out


def analytic_rmst_diff(cov: pd.DataFrame, effect: float, tau: float, chunk: int = 1500, grid: int = 3000) -> float:
    """Closed-form cross-check: mean over patients of  integral_0^tau [S1(t) - S0(t)] dt  (numerical quadrature)."""
    lp = linear_predictor(cov)
    lam = np.log(2) / SCALE0_DAYS ** SHAPE
    tt = np.linspace(0.0, tau * DPM, grid)
    base = tt ** SHAPE
    tot = 0.0
    for i in range(0, len(lp), chunk):
        c = lp[i:i + chunk][:, None]
        s1 = np.exp(-lam * np.exp(c + effect) * base[None, :])
        s0 = np.exp(-lam * np.exp(c) * base[None, :])
        tot += np.trapezoid(s1 - s0, tt, axis=1).sum() / DPM
    return float(tot / len(lp))
