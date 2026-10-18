"""Phase 1 tests, organised by what they establish (software correctness, data integrity, statistical validity)."""
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time

from oncrwd import generate, metrics, rmst, truth
from oncrwd import config as C

# SHA-256 of the raw tables + truth tables of generate(n=800, seed=7) recorded from the ORIGINAL generator (before the
# scenario options existed). The default options must reproduce it exactly so earlier results stay reproducible.
BASELINE_SHA256 = "056584f82bbc2a6343c88c4d302867ae893930e300c0e6331085e68232085f6b"


# ----------------------------------------------------------------------------- data integrity
def test_default_options_reproduce_original_generator_exactly():
    d = Path(tempfile.mkdtemp())
    generate.generate(n=800, seed=7, raw_dir=d / "raw", truth_dir=d / "truth")
    h = hashlib.sha256()
    for f in sorted((d / "raw").glob("*.csv")) + [d / "truth" / "truth_patients.csv", d / "truth" / "truth_lines.csv"]:
        h.update(f.read_bytes())
    assert h.hexdigest() == BASELINE_SHA256


def test_io_probability_defaults_and_monotonicity():
    base = lambda e, p: 1 / (1 + np.exp(-(0.9 - 0.5 * e + 0.9 * (p >= 50) - 0.4 * (p < 1))))  # noqa: E731
    for e in range(4):
        for p in (0, 20, 80):
            assert generate._io_prob(e, p, 1.3) == base(e, p)            # defaults ignore lp entirely
    lp_ms = (0.0, 1.0)
    lo = generate._io_prob(1, 20, -1.0, 1.0, 1.0, lp_ms)
    hi = generate._io_prob(1, 20, +1.0, 1.0, 1.0, lp_ms)
    assert lo > hi                                                          # worse prognosis -> less likely IO (confounding)
    wide = [generate._io_prob(e, p, 0.0, 4.0, 0.0, lp_ms) for e in range(4) for p in (0, 20, 80)]
    narrow = [generate._io_prob(e, p, 0.0, 1.0, 0.0, lp_ms) for e in range(4) for p in (0, 20, 80)]
    assert min(wide) < min(narrow) and max(wide) > max(narrow)             # positivity multiplier widens the PS range


def test_scenario_options_are_written_with_the_truth():
    d = Path(tempfile.mkdtemp())
    generate.generate(n=60, seed=3, raw_dir=d / "raw", truth_dir=d / "truth", opts={"io_chemo_effect": 0.0})
    assert '"io_chemo_effect": 0.0' in (d / "truth" / "scenario_opts.json").read_text()


# ----------------------------------------------------------------------------- software correctness
def _surv_data(n=300, seed=0, cens=True):
    rng = np.random.default_rng(seed)
    t = rng.weibull(1.2, n) * 20
    c = rng.exponential(60, n) if cens else np.full(n, np.inf)
    return np.minimum(t, c), (t <= c).astype(int)


@pytest.mark.parametrize("tau", [12.0, 24.0])
def test_km_rmst_matches_lifelines(tau):
    time, ev = _surv_data()
    ours = rmst.km_rmst(time, ev, tau)
    kmf = KaplanMeierFitter().fit(time, ev)
    assert ours == pytest.approx(restricted_mean_survival_time(kmf, t=tau), rel=1e-9)


def test_km_rmst_hand_example():
    # three deaths at 1, 2, 3 months, tau = 3: S = 1 on [0,1), 2/3 on [1,2), 1/3 on [2,3) -> 1 + 2/3 + 1/3 = 2
    assert rmst.km_rmst([1, 2, 3], [1, 1, 1], 3.0) == pytest.approx(2.0)


def test_km_rmst_variance_matches_bootstrap():
    time, ev = _surv_data(400, seed=1)
    rng = np.random.default_rng(5)
    boots = [rmst.km_rmst(time[i], ev[i], 24.0) for i in (rng.integers(0, 400, 400) for _ in range(600))]
    assert np.sqrt(rmst.km_rmst_var(time, ev, 24.0)) == pytest.approx(np.std(boots, ddof=1), rel=0.15)


def test_weighted_km_with_unit_weights_equals_unweighted():
    time, ev = _surv_data()
    assert rmst.km_rmst(time, ev, 24.0, np.ones(len(time))) == pytest.approx(rmst.km_rmst(time, ev, 24.0))


# ----------------------------------------------------------------------------- statistical validity of the truth
def _fake_population(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"age": rng.normal(68, 9, n), "male": rng.random(n) < 0.52, "hist": rng.choice(["adeno", "squam", "other"], n),
                         "stage": rng.choice([3, 4], n), "ecog": rng.integers(0, 4, n), "n_comorb": rng.poisson(1.0, n),
                         "brain": rng.random(n) < 0.1, "egfr": rng.random(n) < 0.1})


def test_truth_counterfactual_matches_analytic_integral():
    cov = _fake_population()
    th = truth.counterfactual_truth(cov, C.TRUE_LOG_HR["io_chemo_vs_chemo"], m=40)
    for tau in (24, 12):
        an = truth.analytic_rmst_diff(cov, C.TRUE_LOG_HR["io_chemo_vs_chemo"], float(tau))
        assert abs(th[f"rmst_diff_{tau}"] - an) < 4 * th[f"rmst_diff_{tau}_mcse"]


def test_truth_under_the_null_is_zero():
    cov = _fake_population(seed=2)
    th = truth.counterfactual_truth(cov, 0.0, m=20)
    assert abs(th["rmst_diff_24"]) < 4 * th["rmst_diff_24_mcse"]
    assert abs(th["marginal_log_hr"]) < 4 * th["marginal_log_hr_se_approx"]


def test_marginal_hr_is_attenuated_relative_to_conditional():
    th = truth.counterfactual_truth(_fake_population(seed=4), -0.7, m=20)
    assert -0.7 < th["marginal_log_hr"] < 0           # non-collapsibility: marginal log-HR is closer to 0


def test_truth_module_does_not_import_estimator_code():
    src = Path(truth.__file__).read_text()
    for bad in ("analysis", "rmst", "cohort", "lot", "generate"):
        assert f"from . import {bad}" not in src and f"import {bad}" not in src.replace("from . import config", "")


# ----------------------------------------------------------------------------- metrics
def test_metrics_formulas_and_failure_handling():
    est = np.array([0.9, 1.1, 1.0, 1.2, 0.8])
    b, bm = metrics.bias(est, 1.0)
    assert b == pytest.approx(0.0, abs=1e-12) and bm == pytest.approx(est.std(ddof=1) / np.sqrt(5))
    sd, sd_mc = metrics.emp_se(est)
    assert sd_mc == pytest.approx(sd / np.sqrt(2 * 4))
    p, p_mc = metrics.proportion([1, 1, 0, 1])
    assert (p, p_mc) == pytest.approx((0.75, np.sqrt(0.75 * 0.25 / 4)))
    rows = pd.DataFrame({"method": ["cox_multivariable"] * 10, "est": [-0.35] * 8 + [np.nan] * 2, "se": [0.1] * 8 + [np.nan] * 2,
                         "lo": [-0.55] * 8 + [np.nan] * 2, "hi": [-0.15] * 8 + [np.nan] * 2,
                         "failure": [""] * 8 + ["no fit"] * 2})
    # two failures out of ten replicates: failure rate 0.2; coverage counts failures as misses (8/10) and 1.0 among successes
    rows.loc[:7, "est"] = np.linspace(-0.36, -0.34, 8)
    s = metrics.summarise(rows, {"conditional_log_hr": -0.35}).iloc[0]
    assert s["fail_rate"] == pytest.approx(0.2) and s["coverage"] == pytest.approx(0.8) and s["coverage_among_ok"] == pytest.approx(1.0)
    assert bool(s["flag_gt5pct_failures"])


def test_each_estimator_has_one_target_truth():
    assert metrics.TARGET["cox_multivariable"] == "conditional_log_hr"
    assert metrics.TARGET["cox_iptw"] == "marginal_log_hr"
    assert all(metrics.TARGET[m] == "rmst_diff_24" for m in metrics.TARGET if m.startswith("rmst24"))


# ----------------------------------------------------------------------------- estimators recover a known answer
def test_rmst_estimators_recover_known_truth_in_a_confounded_toy_problem():
    rng = np.random.default_rng(11)
    n = 5000
    x = rng.normal(size=n)
    a = (rng.random(n) < 1 / (1 + np.exp(-0.8 * x))).astype(int)        # confounded assignment
    lp = 0.5 * x - 0.35 * a
    k, lam = 1.1, np.log(2) / 20.0 ** 1.1
    t = (-np.log(rng.random(n)) / (lam * np.exp(lp))) ** (1 / k)
    c = rng.exponential(60, n)
    d = pd.DataFrame({"os_months": np.minimum(t, c), "event": (t <= c).astype(int), "io": a})
    X = pd.DataFrame({"x": x})
    tau = 24.0
    tt = np.linspace(0, tau, 4001)
    s1 = np.exp(-lam * np.exp(0.5 * x[:, None] - 0.35) * tt[None, :] ** k)
    s0 = np.exp(-lam * np.exp(0.5 * x[:, None]) * tt[None, :] ** k)
    truth_val = float(np.trapezoid(s1 - s0, tt, axis=1).mean())
    km = rmst.rmst_km(d, tau)
    ip = rmst.rmst_iptw(d, X, ["x"], tau, 40, np.random.default_rng(1))
    gc = rmst.rmst_gcomp(d, X, ["x"], tau, 40, np.random.default_rng(2))
    ai = rmst.rmst_aipw(d, X, ["x"], tau)
    for name, r in (("iptw", ip), ("gcomp", gc), ("aipw", ai)):
        assert "failure" not in r, (name, r)
        assert abs(r["estimate"] - truth_val) < 3.5 * r["se"], (name, r, truth_val)
    assert abs(km["estimate"] - truth_val) > abs(gc["estimate"] - truth_val)   # confounding biases the unadjusted estimator more
