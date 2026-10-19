"""Restricted-mean-survival-time (RMST) estimators for the IO+chemo vs chemo contrast (estimand E2, plus E3).

All estimators return a dict {estimate, se, lo, hi} for  RMST(tau | IO+chemo) - RMST(tau | chemo)  in months, or
{"failure": reason} when no estimate / finite SE can be produced. Kaplan-Meier code is written here (numpy) so the
estimators do not depend on lifelines' KM objects and can be bootstrapped cheaply.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

Z = 1.959963984540054


# ----------------------------------------------------------------------------- Kaplan-Meier building blocks
def km_steps(time, event, weights=None):
    """Kaplan-Meier step function. Returns (event_times, S just after each time, n_at_risk, d). Weighted if weights."""
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    w = np.ones_like(time) if weights is None else np.asarray(weights, float)
    ut, inv = np.unique(time, return_inverse=True)
    d = np.bincount(inv, weights=w * event, minlength=len(ut))
    leave = np.bincount(inv, weights=w, minlength=len(ut))
    n_risk = w.sum() - np.cumsum(leave) + leave
    with np.errstate(divide="ignore", invalid="ignore"):
        surv = np.cumprod(np.where(n_risk > 0, 1.0 - d / n_risk, 1.0))
    return ut, surv, n_risk, d


def rmst_from_steps(ut, surv, tau):
    """Integral of the step function S from 0 to tau."""
    knots = np.concatenate([[0.0], ut[ut < tau], [tau]])
    s_before = np.concatenate([[1.0], surv[: len(knots) - 2]])
    return float(np.sum(s_before * np.diff(knots)))


def km_rmst(time, event, tau, weights=None):
    ut, surv, _, _ = km_steps(time, event, weights)
    return rmst_from_steps(ut, surv, tau)


def km_rmst_var(time, event, tau):
    """Greenwood-type variance of the unweighted KM RMST: sum_i (area under S from t_i to tau)^2 d_i / (n_i (n_i - d_i))."""
    ut, surv, n_risk, d = km_steps(time, event)
    keep = ut < tau
    ut_k, surv_k, n_k, d_k = ut[keep], surv[keep], n_risk[keep], d[keep]
    if len(ut_k) == 0:
        return 0.0
    # area under S from knot i to tau: S is constant (= surv_k[j]) on [ut_k[j], ut_k[j+1]) and surv_k[-1] up to tau
    edges = np.concatenate([ut_k, [tau]])
    seg = surv_k * np.diff(edges)                       # integral of S over [ut_k[j], ut_k[j+1])
    areas = np.cumsum(seg[::-1])[::-1]                  # sum over j >= i
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where((n_k - d_k) > 0, areas ** 2 * d_k / (n_k * (n_k - d_k)), 0.0)
    return float(terms.sum())


def _interval(est, se):
    return {"estimate": est, "se": se, "lo": est - Z * se, "hi": est + Z * se}


def _check(est, se):
    if not (np.isfinite(est) and np.isfinite(se) and se > 0):
        return {"failure": f"non-finite estimate or se (est={est}, se={se})"}
    return _interval(est, se)


# ----------------------------------------------------------------------------- 1. unadjusted KM
def rmst_km(d: pd.DataFrame, tau: float, treat="io", time="os_months", event="event"):
    a, b = d[d[treat] == 1], d[d[treat] == 0]
    if len(a) < 2 or len(b) < 2:
        return {"failure": "empty arm"}
    est = km_rmst(a[time], a[event], tau) - km_rmst(b[time], b[event], tau)
    se = np.sqrt(km_rmst_var(a[time], a[event], tau) + km_rmst_var(b[time], b[event], tau))
    return _check(est, se)


# ----------------------------------------------------------------------------- 2. IPTW KM
def ps_weights(d, covs, treat="io", trim=(0.01, 0.99)):
    """Stabilised ATE weights from an unpenalised logistic model on design-matrix columns; trimmed at 1/99 percentiles.
    Same construction as analysis.propensity_weights, but takes an already-built design matrix."""
    from sklearn.linear_model import LogisticRegression
    X = d[covs]
    Xs = (X - X.mean()) / X.std().replace(0, 1)
    lr = LogisticRegression(penalty=None, max_iter=2000).fit(Xs, d[treat])
    ps = pd.Series(lr.predict_proba(Xs)[:, 1], index=d.index)
    p_t = d[treat].mean()
    w = pd.Series(np.where(d[treat] == 1, p_t / ps, (1 - p_t) / (1 - ps)), index=d.index)
    return ps, w.clip(w.quantile(trim[0]), w.quantile(trim[1]))


def _iptw_point(d, covs, tau, treat="io", time="os_months", event="event"):
    ps, w = ps_weights(d, covs, treat)
    a, b = (d[treat] == 1).values, (d[treat] == 0).values
    return (km_rmst(d[time].values[a], d[event].values[a], tau, w.values[a])
            - km_rmst(d[time].values[b], d[event].values[b], tau, w.values[b]))


def _boot(point_fn, d, B, rng):
    vals = []
    n = len(d)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        bd = d.iloc[idx].reset_index(drop=True)
        if bd["io"].nunique() < 2:
            continue
        try:
            vals.append(point_fn(bd))
        except Exception:
            continue
    return np.array(vals)


def rmst_iptw(d, X, covs, tau, B, rng, treat="io"):
    """IPTW-weighted KM RMST difference; propensity model re-fitted in every bootstrap resample; normal-approx interval."""
    dd = pd.concat([d[["os_months", "event", treat]], X[covs]], axis=1).reset_index(drop=True)
    try:
        est = _iptw_point(dd, covs, tau)
        boots = _boot(lambda bd: _iptw_point(bd, covs, tau), dd, B, rng)
    except Exception as e:  # noqa: BLE001
        return {"failure": f"iptw: {e}"}
    if len(boots) < 0.8 * B:
        return {"failure": f"iptw: only {len(boots)}/{B} bootstrap resamples valid"}
    return _check(est, float(boots.std(ddof=1)))


# ----------------------------------------------------------------------------- 3. g-computation (Cox outcome model)
def _cox_rmst_matrix(cph, Xrows: pd.DataFrame, tau):
    """Per-patient RMST(tau) from a fitted Cox model: integral of exp(-H0(t) exp(x beta)) dt."""
    bh = cph.baseline_cumulative_hazard_.iloc[:, 0]
    t = bh.index.values.astype(float)
    t = t[t < tau]
    H = bh.loc[bh.index[: len(t)]].values
    lin = cph.predict_log_partial_hazard(Xrows).values  # relative to the model's centering
    # lifelines centres covariates: baseline hazard refers to mean covariates, predict_log_partial_hazard is consistent with it
    knots = np.concatenate([[0.0], t, [tau]])
    H_before = np.concatenate([[0.0], H])
    S = np.exp(-np.outer(np.exp(lin), H_before))
    return (S * np.diff(knots)[None, :]).sum(axis=1)


def _gcomp_point(dd, covs, tau):
    cols = ["io"] + covs
    cph = CoxPHFitter().fit(dd[["os_months", "event"] + cols], "os_months", "event")
    X1, X0 = dd[cols].copy(), dd[cols].copy()
    X1["io"], X0["io"] = 1, 0
    return float(_cox_rmst_matrix(cph, X1, tau).mean() - _cox_rmst_matrix(cph, X0, tau).mean()), cph


def rmst_gcomp(d, X, covs, tau, B, rng):
    dd = pd.concat([d[["os_months", "event", "io"]], X[covs]], axis=1).reset_index(drop=True)
    try:
        est, _ = _gcomp_point(dd, covs, tau)
        boots = _boot(lambda bd: _gcomp_point(bd, covs, tau)[0], dd, B, rng)
    except Exception as e:  # noqa: BLE001
        return {"failure": f"gcomp: {e}"}
    if len(boots) < 0.8 * B:
        return {"failure": f"gcomp: only {len(boots)}/{B} bootstrap resamples valid"}
    return _check(est, float(boots.std(ddof=1)))


# ----------------------------------------------------------------------------- 4. AIPW (IPCW) with plug-in variance
def rmst_aipw(d, X, covs, tau):
    """Doubly robust RMST difference. Outcome model: Cox. Propensity: logistic (as IPTW). Censoring: marginal KM IPCW.

    psi_a = mu_a(X) + 1(A=a)/pi_a(X) * W * (Y - mu_a(X)),  Y = min(T, tau), W = R / G(min(U, tau)-),
    R = 1 if death observed before tau or follow-up reaches tau. SE is the plug-in sd(psi)/sqrt(n): nuisance-estimation
    uncertainty is ignored, so the model-SE ratio in the simulation is the honest check on it.
    """
    dd = pd.concat([d[["os_months", "event", "io"]], X[covs]], axis=1).reset_index(drop=True)
    try:
        _, cph = _gcomp_point(dd, covs, tau)
        cols = ["io"] + covs
        X1, X0 = dd[cols].copy(), dd[cols].copy()
        X1["io"], X0["io"] = 1, 0
        mu1, mu0 = _cox_rmst_matrix(cph, X1, tau), _cox_rmst_matrix(cph, X0, tau)
        ps, _ = ps_weights(dd, covs)
        pi1 = ps.clip(0.01, 0.99).values
        U, ev = dd["os_months"].values, dd["event"].values
        Y = np.minimum(U, tau)
        R = ((ev == 1) & (U < tau)) | (U >= tau)
        ut, G, _, _ = km_steps(U, 1 - ev)                           # censoring survival function
        idx = np.searchsorted(ut, np.minimum(U, tau), side="left") - 1  # G just before min(U, tau)
        Gm = np.where(idx >= 0, G[np.clip(idx, 0, None)], 1.0)
        W = np.where(R, 1.0 / np.clip(Gm, 1e-3, None), 0.0)
        a = dd["io"].values
        psi1 = mu1 + a / pi1 * W * (Y - mu1)
        psi0 = mu0 + (1 - a) / (1 - pi1) * W * (Y - mu0)
        psi = psi1 - psi0
        est = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    except Exception as e:  # noqa: BLE001
        return {"failure": f"aipw: {e}"}
    return _check(est, se)


# ----------------------------------------------------------------------------- 5. survival probability difference (E3)
def surv_diff_km(d, t_point, treat="io"):
    """Unadjusted KM survival-probability difference at t_point with Greenwood variance (secondary estimand E3)."""
    out = []
    for g in (1, 0):
        x = d[d[treat] == g]
        ut, surv, n_risk, dd_ = km_steps(x["os_months"], x["event"])
        k = ut <= t_point
        s = surv[k][-1] if k.any() else 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            gw = np.sum(np.where((n_risk[k] - dd_[k]) > 0, dd_[k] / (n_risk[k] * (n_risk[k] - dd_[k])), 0.0))
        out.append((s, s ** 2 * gw))
    est = out[0][0] - out[1][0]
    return _check(est, float(np.sqrt(out[0][1] + out[1][1])))
