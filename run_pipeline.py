"""End-to-end pipeline: generate -> QC -> LOT -> cohort -> survival analysis -> report.

    python run_pipeline.py            # reuse data/raw if present
    python run_pipeline.py --regen    # regenerate the synthetic EHR first
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "src"))
from oncrwd import analysis as A  # noqa: E402
from oncrwd import cohort as K  # noqa: E402
from oncrwd import config as C  # noqa: E402
from oncrwd import generate, lot, plots  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def md(df: pd.DataFrame, floatfmt="{:.2f}", index=True) -> str:
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    body = ""
    for _, r in d.iterrows():
        body += "| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |\n"
    return head + body


def table1(coh: pd.DataFrame) -> pd.DataFrame:
    order = ["Platinum chemo", "IO + chemo", "IO mono", "EGFR TKI"]
    g = coh.groupby("regimen_class")
    rows = {"N": g.size()}
    rows["Age, mean (SD)"] = g["age"].agg(lambda s: f"{s.mean():.1f} ({s.std():.1f})")
    rows["Male, %"] = g["male"].mean().mul(100).round(0)
    rows["Stage IV, %"] = g["stage"].apply(lambda s: (s == 4).mean() * 100).round(0)
    for lvl in ["0", "1", "2", "3+", "Unknown"]:
        rows[f"ECOG {lvl}, %"] = g["ecog"].apply(lambda s, l=lvl: (s == l).mean() * 100).round(0)
    for lvl in ["<1%", "1-49%", ">=50%", "Unknown"]:
        rows[f"PD-L1 {lvl}, %"] = g["pdl1"].apply(lambda s, l=lvl: (s == l).mean() * 100).round(0)
    rows["EGFR positive, %"] = g["egfr"].apply(lambda s: (s == "Positive").mean() * 100).round(0)
    rows["Squamous, %"] = g["histology"].apply(lambda s: (s == "Squamous").mean() * 100).round(0)
    rows["Brain mets, %"] = g["brain_mets"].mean().mul(100).round(0)
    rows["Comorbidities, mean"] = g["n_comorbid"].mean().round(2)
    return pd.DataFrame(rows).T[[c for c in order if c in coh["regimen_class"].unique()]]


def main(regen: bool = False):
    C.RESULTS_DIR.mkdir(exist_ok=True)
    if regen or not (C.RAW_DIR / "patients.csv").exists():
        print("Generating synthetic EHR ...")
        generate.generate()
    summary: dict = {}

    # ---------------------------------------------------------------- 1. QC
    raw = K.load_raw()
    clean, qc = K.run_qc(raw)
    qc.to_csv(C.RESULTS_DIR / "qc_log.csv", index=False)
    summary["raw_rows"] = {k: len(v) for k, v in raw.items()}

    # ---------------------------------------------------------------- 2. Line of therapy
    admin = lot.build_admin_table(clean["medications"])
    lines = lot.derive_lines(admin)
    lines.to_csv(C.RESULTS_DIR / "derived_lines_of_therapy.csv", index=False)
    truth = pd.read_csv(C.TRUTH_DIR / "truth_lines.csv", parse_dates=["start", "end"])
    tpat = pd.read_csv(C.TRUTH_DIR / "truth_patients.csv")
    valid = tpat.loc[~tpat["corrupted"], "patient_id"]
    summary["lot_validation"] = lot.evaluate_against_truth(lines, truth, valid)
    grid = lot.sensitivity_grid(admin, truth, valid)
    grid.to_csv(C.RESULTS_DIR / "lot_sensitivity_grid.csv", index=False)
    plots.lot_sensitivity(grid)

    # ---------------------------------------------------------------- 3. Cohort
    coh, att = K.build_cohort(clean, lines)
    coh.to_csv(C.RESULTS_DIR / "analytic_cohort.csv")
    att.to_csv(C.RESULTS_DIR / "attrition.csv", index=False)
    plots.attrition(att)
    plots.swimmer(lines, coh)
    trans = plots.transitions(lines, coh.index)
    summary["cohort_n"] = int(len(coh))
    t1 = table1(coh)
    t1.to_csv(C.RESULTS_DIR / "table1.csv")

    # ---------------------------------------------------------------- 4. KM: OS and TTNT by first-line regimen class
    order = ["Platinum chemo", "IO + chemo", "IO mono", "EGFR TKI"]
    km_os = A.km_table(coh, "regimen_class")
    km_os.to_csv(C.RESULTS_DIR / "km_os_by_regimen.csv", index=False)
    summary["km_os_logrank_p"] = km_os.attrs["logrank_p"]
    km_tt = A.km_table(coh, "regimen_class", time="ttnt_months", event="ttnt_event")
    km_tt.to_csv(C.RESULTS_DIR / "km_ttnt_by_regimen.csv", index=False)
    plots.km_by_group(coh, "regimen_class", "os_months", "event", "Overall survival by first-line regimen (all treated advanced NSCLC)",
                      "Months since first-line start", "02_km_os_regimen.png", order=order)
    plots.km_by_group(coh, "regimen_class", "ttnt_months", "ttnt_event", "Time to next treatment or death (TTNT)",
                      "Months since first-line start", "03_km_ttnt_regimen.png", order=order, tmax=36)

    # ---------------------------------------------------------------- 5. Comparative effectiveness: IO+chemo vs chemo
    era = pd.Timestamp(C.IO_ERA_START)
    cmp_ = coh[(coh["index_date"] >= era) & (coh["egfr"] != "Positive")
               & coh["regimen_class"].isin(["IO + chemo", "Platinum chemo"])].copy()
    cmp_["io"] = (cmp_["regimen_class"] == "IO + chemo").astype(int)
    summary["comparative_cohort"] = {"n": int(len(cmp_)), "n_io_chemo": int(cmp_["io"].sum()),
                                     "n_chemo": int((1 - cmp_["io"]).sum()), "events": int(cmp_["event"].sum())}
    X = A.make_X(cmp_, age_scale=10)
    covs = A.varying(X)
    d = pd.concat([cmp_[["os_months", "event", "io"]], X[covs]], axis=1)

    unadj = A.cox_fit(d, ["io"])
    adj = A.cox_fit(d, ["io"] + covs)
    ps, w = A.propensity_weights(cmp_, "io", A.varying(A.make_X(cmp_)))
    d["w"] = w
    iptw = A.cox_fit(d, ["io"], weights="w")
    res = {}
    for name, m in [("Unadjusted", unadj), ("Multivariable Cox", adj), ("IPTW (stabilised, trimmed 1/99)", iptw)]:
        s = m.summary.loc["io"]
        res[name] = (s["exp(coef)"], s["exp(coef) lower 95%"], s["exp(coef) upper 95%"], s["coef"])
    comp = pd.DataFrame(res, index=["hr", "hr_low", "hr_high", "log_hr"]).T
    comp["true_hr"] = np.exp(C.TRUE_LOG_HR["io_chemo_vs_chemo"])
    comp.to_csv(C.RESULTS_DIR / "io_vs_chemo_estimates.csv")
    summary["io_vs_chemo"] = comp.round(3).to_dict("index")
    summary["e_value_multivariable"] = A.e_value(comp.loc["Multivariable Cox", "hr"], comp.loc["Multivariable Cox", "hr_high"])

    xc = A.varying(A.make_X(cmp_))
    smd_b, smd_a = A.smd_table(cmp_, "io", xc), A.smd_table(cmp_, "io", xc, w)
    summary["max_abs_smd"] = {"before": float(smd_b.abs().max()), "after": float(smd_a.abs().max())}
    plots.love_plot(smd_b, smd_a)
    plots.weighted_km_plot(A.weighted_km(cmp_, "io", w))
    summary["ps_overlap"] = {"ps_min": float(ps.min()), "ps_max": float(ps.max()), "weight_max": float(w.max())}

    ph = A.ph_test(adj, d, ["io"] + covs)
    ph.to_csv(C.RESULTS_DIR / "ph_test_adjusted.csv")
    summary["ph_test_min_p"] = float(ph["p"].min())

    labels = {"io": "IO + chemo vs chemo", "age": "Age (per 10 y)", "male": "Male", "stage4": "Stage IV (vs III)", "squamous": "Squamous",
              "brain_mets": "Brain metastases", "n_comorbid": "Comorbidities (per condition)", "ecog1": "ECOG 1 (vs 0)",
              "ecog2": "ECOG 2", "ecog3plus": "ECOG 3+", "ecog_unk": "ECOG unknown", "pdl1_1_49": "PD-L1 1-49% (vs <1%)",
              "pdl1_ge50": "PD-L1 ≥50%", "pdl1_unk": "PD-L1 unknown", "smoke_former": "Former smoker (vs never)",
              "smoke_current": "Current smoker", "smoke_unk": "Smoking unknown"}
    plots.forest(A.cox_summary(adj), "Multivariable Cox model — OS, IO + chemo vs chemo", "04_forest_adjusted_cox.png", labels)

    # sensitivity: 3-arm incl. IO mono
    c3 = coh[(coh["index_date"] >= era) & (coh["egfr"] != "Positive") & coh["regimen_class"].isin(["IO + chemo", "IO mono", "Platinum chemo"])].copy()
    c3["io_chemo"] = (c3["regimen_class"] == "IO + chemo").astype(int)
    c3["io_mono"] = (c3["regimen_class"] == "IO mono").astype(int)
    X3 = A.make_X(c3, age_scale=10)
    c3x = A.varying(X3)
    d3 = pd.concat([c3[["os_months", "event", "io_chemo", "io_mono"]], X3[c3x]], axis=1)
    m3 = A.cox_fit(d3, ["io_chemo", "io_mono"] + c3x)
    summary["three_arm"] = A.cox_summary(m3).loc[["io_chemo", "io_mono"]].round(3).to_dict("index")

    # ---------------------------------------------------------------- 6. Truth recovery in the full treated cohort
    Xf = A.make_X(coh)
    f = pd.concat([coh[["os_months", "event"]], Xf], axis=1)
    f["ecog_num"] = coh["ecog"].map({"0": 0, "1": 1, "2": 2, "3+": 3}).fillna(1.0)
    f["IO_chemo"] = (coh["regimen_class"] == "IO + chemo").astype(int)
    f["IO_mono"] = (coh["regimen_class"] == "IO mono").astype(int)
    f["EGFR_TKI"] = (coh["regimen_class"] == "EGFR TKI").astype(int)
    fcols = ["IO_chemo", "IO_mono", "EGFR_TKI", "age", "ecog_num", "ecog_unk", "stage4", "brain_mets", "n_comorbid", "male", "squamous", "egfr_pos"]
    full = A.cox_fit(f, fcols)
    fs = A.cox_summary(full)
    T = C.TRUE_LOG_HR
    truth_map = {"IO_chemo": T["io_chemo_vs_chemo"], "IO_mono": T["io_mono_vs_chemo"], "age": T["age_per_year"],
                 "ecog_num": T["ecog_per_level"], "stage4": T["stage4_vs_3"], "brain_mets": T["brain_mets"],
                 "n_comorbid": T["comorbidity_per_condition"], "male": T["male"], "squamous": T["squamous"]}
    rec = pd.DataFrame({"est_log_hr": fs.loc[list(truth_map), "log_hr"], "lo": np.log(fs.loc[list(truth_map), "hr_low"]),
                        "hi": np.log(fs.loc[list(truth_map), "hr_high"]), "true_log_hr": pd.Series(truth_map)})
    rec["covered"] = (rec["lo"] <= rec["true_log_hr"]) & (rec["true_log_hr"] <= rec["hi"])
    rec.rename(index={"IO_chemo": "IO + chemo vs chemo", "IO_mono": "IO mono vs chemo", "age": "Age (per year)", "ecog_num": "ECOG (per level)",
                      "stage4": "Stage IV vs III", "brain_mets": "Brain mets", "n_comorbid": "Comorbidity (each)", "male": "Male",
                      "squamous": "Squamous"}, inplace=True)
    rec.to_csv(C.RESULTS_DIR / "truth_recovery.csv")
    plots.truth_recovery(rec)
    summary["truth_recovery_coverage"] = f"{int(rec['covered'].sum())}/{len(rec)}"

    # ---------------------------------------------------------------- 7. Immortal-time bias
    itb = A.immortal_time_demo(coh)
    itb.to_csv(C.RESULTS_DIR / "immortal_time_bias.csv", index=False)
    plots.immortal(itb)
    summary["immortal_time"] = itb.round(3).to_dict("records")

    (C.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---------------------------------------------------------------- 8. Report
    L = summary["lot_validation"]
    rep = f"""# Results — oncology RWD pipeline (synthetic NSCLC EHR)

*Auto-generated by `run_pipeline.py`. All data are synthetic.*

## 1. Data quality log
{md(qc, index=False)}
## 2. Cohort attrition
{md(att, index=False)}
## 3. Line-of-therapy derivation vs simulated truth
n = {L['patients_compared']:,} patients (mis-dated records excluded)

| Metric | Value |
|---|---|
| Whole-sequence exact match | {L['sequence_exact_match']:.1%} |
| Number of lines agrees | {L['n_lines_agreement']:.1%} |
| Line-1 regimen correct | {L['line1_regimen_accuracy']:.1%} |
| Line start within 7 d | {L['line_start_within_7d']:.1%} |

Sensitivity (exact sequence match) — see `lot_sensitivity_grid.csv`:

{md(grid.pivot(index='window_days', columns='gap_days', values='sequence_exact_match'), floatfmt="{:.3f}")}
## 4. Baseline characteristics by first-line regimen (n = {len(coh):,})
{md(t1, floatfmt="{:.1f}")}
## 5. Overall survival (months from first-line start)
Log-rank p = {km_os.attrs['logrank_p']:.2g}

{md(km_os.drop(columns=['median_followup_months']), index=False)}
## 6. Time to next treatment or death
{md(km_tt.drop(columns=['median_followup_months']), index=False)}
## 7. IO + chemo vs platinum chemo (EGFR-negative/unknown, index ≥ {C.IO_ERA_START}; n = {len(cmp_):,})
{md(comp)}
Max |SMD| before/after weighting: {smd_b.abs().max():.2f} → {smd_a.abs().max():.2f}.  Min PH-test p across covariates: {ph['p'].min():.3f}.
E-value (multivariable HR): {summary['e_value_multivariable']}

## 8. Truth recovery (full treated cohort Cox)
{md(rec)}
## 9. Immortal-time bias (true HR = 1)
{md(itb, index=False)}
"""
    (C.RESULTS_DIR / "report.md").write_text(rep, encoding="utf-8")
    print(f"Report written to {C.RESULTS_DIR / 'report.md'}")
    return summary


if __name__ == "__main__":
    main(regen="--regen" in sys.argv)
