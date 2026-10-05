"""Integration test: generate a small EHR, run QC -> LOT -> cohort -> survival models."""
import pandas as pd
import pytest

from oncrwd import analysis as A
from oncrwd import cohort as K
from oncrwd import generate, lot


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    generate.generate(n=1500, seed=11, raw_dir=root / "raw", truth_dir=root / "truth")
    clean, qc = K.run_qc(K.load_raw(root / "raw"))
    admin = lot.build_admin_table(clean["medications"])
    lines = lot.derive_lines(admin)
    coh, att = K.build_cohort(clean, lines)
    truth = pd.read_csv(root / "truth" / "truth_lines.csv", parse_dates=["start", "end"])
    tp = pd.read_csv(root / "truth" / "truth_patients.csv")
    return dict(clean=clean, qc=qc, admin=admin, lines=lines, coh=coh, att=att, truth=truth, tp=tp)


def test_generator_writes_synthea_style_tables(built):
    assert {"Id", "BIRTHDATE", "DEATHDATE", "GENDER"} <= set(built["clean"]["patients"].columns)
    assert {"START", "STOP", "PATIENT", "CODE", "DESCRIPTION"} <= set(built["clean"]["medications"].columns)


def test_qc_removes_duplicates(built):
    row = built["qc"].set_index("check").loc["Exact duplicate medication rows"]
    assert row["n_flagged"] > 0
    assert not built["clean"]["medications"].duplicated().any()


def test_attrition_is_monotone_and_consistent(built):
    att = built["att"]
    assert att["n_remaining"].is_monotonic_decreasing
    assert att["n_remaining"].iloc[-1] == len(built["coh"])
    assert (att["n_excluded"] >= 0).all()


def test_cohort_integrity(built):
    c = built["coh"]
    assert (c["os_months"] > 0).all()
    assert (c["index_date"] >= c["dx_date"]).all()
    assert (c["line1_start_to_dx"] <= 90).all()
    assert c["stage"].isin([3, 4]).all() and (c["age"] >= 18).all()
    assert c["event"].isin([0, 1]).all()
    assert (c["ttnt_months"] <= c["os_months"] + 1e-9).all()      # next-line start/death can never follow end of follow-up


def test_lot_recovers_simulated_truth(built):
    valid = built["tp"].loc[~built["tp"]["corrupted"], "patient_id"]
    ev = lot.evaluate_against_truth(built["lines"], built["truth"], valid)
    assert ev["sequence_exact_match"] > 0.93
    assert ev["line1_regimen_accuracy"] > 0.95


def test_cox_runs_and_direction_is_plausible(built):
    c = built["coh"]
    X = A.make_X(c)
    cols = A.varying(X)
    d = pd.concat([c[["os_months", "event"]], X[cols]], axis=1)
    s = A.cox_summary(A.cox_fit(d, cols))
    assert s.loc["stage4", "hr"] > 1 and s.loc["age", "hr"] > 1


def test_immortal_time_naive_is_biased_downward(built):
    itb = A.immortal_time_demo(built["coh"]).set_index("method")
    assert itb.loc["Naive (ever-treated, from index)", "hr"] < 0.8
    assert itb.loc["Time-varying exposure Cox", "hr"] > itb.loc["Naive (ever-treated, from index)", "hr"]


def test_e_value_properties():
    assert A.e_value(0.7)["e_value_point"] > 1
    assert A.e_value(0.5)["e_value_point"] > A.e_value(0.7)["e_value_point"]
    assert A.e_value(0.7, 1.05)["e_value_ci"] == 1.0
    assert A.e_value(1.0)["e_value_point"] == 1.0


def test_iptw_balances_covariates(built):
    c = built["coh"]
    c = c[c["regimen_class"].isin(["IO + chemo", "Platinum chemo"]) & (c["egfr"] != "Positive")].copy()
    c["io"] = (c["regimen_class"] == "IO + chemo").astype(int)
    cols = A.varying(A.make_X(c))
    ps, w = A.propensity_weights(c, "io", cols)
    before, after = A.smd_table(c, "io", cols), A.smd_table(c, "io", cols, w)
    assert after.abs().max() < before.abs().max() and after.abs().max() < 0.1
