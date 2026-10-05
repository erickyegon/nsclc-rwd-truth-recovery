"""Survival analysis: KM / RMST, Cox, IPTW (propensity), E-value, immortal-time-bias demonstration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import median_survival_times, restricted_mean_survival_time
from sklearn.linear_model import LogisticRegression

from . import config as C


# ----------------------------------------------------------------------------- design matrix
def make_X(df: pd.DataFrame, age_scale: float = 1.0) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"] / age_scale
    X["male"] = df["male"]
    X["stage4"] = (df["stage"] == 4).astype(int)
    X["squamous"] = (df["histology"] == "Squamous").astype(int)
    X["brain_mets"] = df["brain_mets"]
    X["n_comorbid"] = df["n_comorbid"]
    for lvl, nm in [("1", "ecog1"), ("2", "ecog2"), ("3+", "ecog3plus"), ("Unknown", "ecog_unk")]:
        X[nm] = (df["ecog"] == lvl).astype(int)
    for lvl, nm in [("1-49%", "pdl1_1_49"), (">=50%", "pdl1_ge50"), ("Unknown", "pdl1_unk")]:
        X[nm] = (df["pdl1"] == lvl).astype(int)
    for lvl, nm in [("Former", "smoke_former"), ("Current", "smoke_current"), ("Unknown", "smoke_unk")]:
        X[nm] = (df["smoking"] == lvl).astype(int)
    X["egfr_pos"] = (df["egfr"] == "Positive").astype(int)
    return X


def varying(X: pd.DataFrame) -> list[str]:
    """Columns with non-zero variance (constant dummies break Cox / propensity fits)."""
    return [c for c in X.columns if X[c].std() > 0]


# ----------------------------------------------------------------------------- Kaplan-Meier
def km_table(df, group, time="os_months", event="event", tau=24.0) -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(group):
        kmf = KaplanMeierFitter().fit(d[time], d[event])
        ci = median_survival_times(kmf.confidence_interval_)
        rows.append({
            group: g, "n": len(d), "events": int(d[event].sum()),
            "median_months": kmf.median_survival_time_,
            "median_ci_low": float(ci.iloc[0, 0]), "median_ci_high": float(ci.iloc[0, 1]),
            "surv_12m": float(kmf.predict(12.0)), "surv_24m": float(kmf.predict(24.0)),
            "rmst_24m": restricted_mean_survival_time(kmf, t=tau),
            "median_followup_months": float(KaplanMeierFitter().fit(d[time], 1 - d[event]).median_survival_time_),
        })
    out = pd.DataFrame(rows)
    lr = multivariate_logrank_test(df[time], df[group], df[event])
    out.attrs["logrank_p"] = float(lr.p_value)
    return out


# ----------------------------------------------------------------------------- Cox
def cox_fit(df, cols, time="os_months", event="event", weights=None, robust=False) -> CoxPHFitter:
    keep = [time, event] + cols + ([weights] if weights else [])
    cph = CoxPHFitter()
    cph.fit(df[keep], time, event, weights_col=weights, robust=robust or weights is not None)
    return cph


def cox_summary(cph) -> pd.DataFrame:
    s = cph.summary[["coef", "se(coef)", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    s.columns = ["log_hr", "se", "hr", "hr_low", "hr_high", "p"]
    return s


def ph_test(cph, df, cols, time="os_months", event="event") -> pd.DataFrame:
    res = proportional_hazard_test(cph, df[[time, event] + cols], time_transform="rank")
    return res.summary


# ----------------------------------------------------------------------------- IPTW
def propensity_weights(df, treat, X_cols, trim=(0.01, 0.99)):
    """Stabilised ATE weights from an unpenalised logistic propensity model; weights truncated at percentiles."""
    X = make_X(df, age_scale=1.0)[X_cols]
    Xs = (X - X.mean()) / X.std().replace(0, 1)
    lr = LogisticRegression(penalty=None, max_iter=2000).fit(Xs, df[treat])
    ps = pd.Series(lr.predict_proba(Xs)[:, 1], index=df.index)
    p_t = df[treat].mean()
    w = np.where(df[treat] == 1, p_t / ps, (1 - p_t) / (1 - ps))
    w = pd.Series(w, index=df.index)
    lo, hi = w.quantile(trim[0]), w.quantile(trim[1])
    return ps, w.clip(lo, hi)


def smd_table(df, treat, X_cols, w=None) -> pd.DataFrame:
    X = make_X(df)[X_cols]
    w = pd.Series(1.0, index=df.index) if w is None else w
    out = []
    for c in X_cols:
        m, v = {}, {}
        for g in (0, 1):
            msk = df[treat] == g
            ww = w[msk]
            m[g] = np.average(X.loc[msk, c], weights=ww)
            v[g] = np.average((X.loc[msk, c] - m[g]) ** 2, weights=ww)
        sd = np.sqrt((v[0] + v[1]) / 2)
        out.append((c, (m[1] - m[0]) / sd if sd > 0 else 0.0))
    return pd.DataFrame(out, columns=["covariate", "smd"]).set_index("covariate")["smd"]


def weighted_km(df, treat, w, time="os_months", event="event"):
    fits = {}
    for g in (0, 1):
        m = df[treat] == g
        fits[g] = KaplanMeierFitter().fit(df.loc[m, time], df.loc[m, event], weights=w[m])
    return fits


def e_value(hr: float, hr_ci_limit: float | None = None) -> dict:
    """VanderWeele & Ding E-value; common-outcome HR->RR approximation."""
    def rr(h):
        h = 1 / h if h < 1 else h
        return (1 - 0.5 ** np.sqrt(h)) / (1 - 0.5 ** np.sqrt(1 / h))

    def ev(r):
        return r + np.sqrt(r * (r - 1)) if r > 1 else 1.0
    out = {"e_value_point": float(ev(rr(hr)))}
    if hr_ci_limit is not None:
        crosses = (hr < 1 and hr_ci_limit >= 1) or (hr > 1 and hr_ci_limit <= 1)
        out["e_value_ci"] = 1.0 if crosses else float(ev(rr(hr_ci_limit)))
    return out


# ----------------------------------------------------------------------------- immortal time bias
def immortal_time_demo(coh: pd.DataFrame, landmark_months: float = 6.0, adj_cols: list[str] | None = None) -> pd.DataFrame:
    """Effect of *receiving second-line therapy* on OS -- true HR = 1 by construction.

    naive      : exposure = 'ever received 2L', followed from first-line start  (immortal time)
    landmark   : exposure fixed at `landmark_months`, follow-up starts there      (restricted to survivors)
    time-varying: exposure switches on at 2L start                               (correct handling)
    """
    adj_cols = adj_cols or ["age", "male", "stage4", "squamous", "brain_mets", "n_comorbid",
                            "ecog1", "ecog2", "ecog3plus", "ecog_unk", "egfr_pos"]
    X = make_X(coh)[adj_cols]
    base = pd.concat([coh[["os_months", "event", "second_line_months"]], X], axis=1)
    # first-line regimen class as an adjustment as well
    for r in ["IO + chemo", "IO mono", "EGFR TKI"]:
        base["l1_" + r.replace(" + ", "_").replace(" ", "_")] = (coh["regimen_class"] == r).astype(int)
    adj = [c for c in base.columns if c not in {"os_months", "event", "second_line_months"}]
    rows = []

    naive = base.copy()
    naive["got_2L"] = naive["second_line_months"].notna().astype(int)
    c = cox_fit(naive, ["got_2L"] + adj)
    rows.append(("Naive (ever-treated, from index)", c))

    lm = base[base["os_months"] > landmark_months].copy()
    lm["got_2L"] = (lm["second_line_months"] <= landmark_months).astype(int)
    lm["os_months"] = lm["os_months"] - landmark_months
    c2 = cox_fit(lm, ["got_2L"] + adj)
    rows.append((f"Landmark ({landmark_months:g} months)", c2))

    b = base.rename_axis("patient_id").reset_index()
    recs = []
    for r in b.itertuples(index=False):
        d = r._asdict()
        t2, end = d["second_line_months"], d["os_months"]
        covs = {k: d[k] for k in adj}
        if pd.notna(t2) and 0 < t2 < end:
            recs.append({"patient_id": d["patient_id"], "start": 0.0, "stop": t2, "event": 0, "got_2L": 0, **covs})
            recs.append({"patient_id": d["patient_id"], "start": t2, "stop": end, "event": d["event"], "got_2L": 1, **covs})
        else:
            recs.append({"patient_id": d["patient_id"], "start": 0.0, "stop": end, "event": d["event"], "got_2L": 0, **covs})
    tv = pd.DataFrame(recs)
    ctv = CoxTimeVaryingFitter().fit(tv, id_col="patient_id", event_col="event", start_col="start", stop_col="stop")
    s_tv = ctv.summary.loc["got_2L"]

    out = []
    for name, cph in rows:
        s = cph.summary.loc["got_2L"]
        out.append((name, s["exp(coef)"], s["exp(coef) lower 95%"], s["exp(coef) upper 95%"]))
    out.append(("Time-varying exposure Cox", s_tv["exp(coef)"], s_tv["exp(coef) lower 95%"], s_tv["exp(coef) upper 95%"]))
    return pd.DataFrame(out, columns=["method", "hr", "hr_low", "hr_high"])
