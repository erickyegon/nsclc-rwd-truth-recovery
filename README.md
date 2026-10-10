# Oncology RWD pipeline — advanced NSCLC (synthetic EHR)

[![tests](https://github.com/erickyegon/nsclc-rwd-truth-recovery/actions/workflows/ci.yml/badge.svg)](https://github.com/erickyegon/nsclc-rwd-truth-recovery/actions/workflows/ci.yml) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

An end-to-end real-world-data (RWD) study on **synthetic oncology EHR data**:
messy EHR tables → QC → **cohort definition** → rule-based **line-of-therapy (LOT) derivation** → **survival analysis**
(Kaplan–Meier, Cox, IPTW, time-varying exposure) — with every estimate checked against a **known simulated truth**.

> **Everything here is synthetic.** No real patients; no clinical conclusion should be drawn from the numbers.
> The point is that the *methods* are demonstrably correct.

## Why a custom generator instead of raw Synthea?
Synthea's lung-cancer module has no systemic-therapy lines, biomarkers or realistic treatment selection, so it cannot
support a line-of-therapy study. `src/oncrwd/generate.py` writes the **same Synthea-style CSV layout** (patients,
conditions, medications, observations, encounters, procedures) but simulates the clinical process explicitly. That also
makes validation possible: the analysis code only sees `data/raw/`, while `data/truth/` holds the true lines,
covariates and hazard ratios.

Built-in real-world mess: duplicate rows · brand/generic/strength drug-name variants · supportive & chronic meds mixed in ·
carboplatin↔cisplatin swaps · late-added second agents · treatment pauses · therapy mis-dated before diagnosis ·
encounters after death · loss to follow-up · missing/mis-recorded ECOG · untested PD-L1/EGFR · administrative censoring.
Treatment choice is confounded by ECOG, PD-L1, EGFR, histology, age and calendar era.

## How this complements the R / OMOP project
This repo is the companion to [**oncology-rwe-nsclc**](https://github.com/erickyegon/oncology-rwe-nsclc) (Synthea → OMOP CDM → dbt/PostgreSQL → R, benchmarked against SEER).

| | [oncology-rwe-nsclc](https://github.com/erickyegon/oncology-rwe-nsclc) | **nsclc-rwd-truth-recovery** (this repo) |
|---|---|---|
| Question it answers | How realistic is off-the-shelf synthetic EHR data? | Is the analysis pipeline itself correct? |
| Data | Synthea, harmonised to OMOP CDM, benchmarked against SEER | Custom generator with a known, recorded truth |
| Stack | R, dbt, PostgreSQL | Python (pandas, lifelines, scikit-learn) |
| Validation | External: real registry survival curves | Internal: truth recovery, 10-seed Monte-Carlo bias and coverage |

The R project shows what happens when synthetic data meets real-world benchmarks; this one shows that the line-of-therapy and survival methods recover a known answer.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Generate | `generate.py` | 6,000-patient Weibull-PH simulation with 4 lines of therapy, biomarkers, noise |
| QC | `cohort.run_qc` | de-duplication, post-death activity, date-integrity checks, logged |
| LOT | `lot.py`, `drugs.py` | drug normalisation, regimen window / gap / maintenance rules (R1–R7), validation + sensitivity grid |
| Cohort | `cohort.build_cohort` | 8-step inclusion/exclusion with attrition, baseline covariates, OS & TTNT endpoints |
| Analysis | `analysis.py` | KM/RMST, Cox + PH test, IPTW + balance, E-value, immortal-time-bias demo |
| Validation | `run_pipeline.py`, `simulation_study.py` | truth recovery, 10-seed Monte-Carlo bias & coverage |

Full protocol-style write-up: [docs/methods.md](docs/methods.md). Auto-generated numbers: `results/report.md`.

## Key results (seed 2026)

**Cohort:** 6,000 EHR patients → 3,655 NSCLC → **1,615** treated advanced NSCLC with first-line therapy ([attrition](results/figures/01_attrition.png)).

**Line-of-therapy validation** (vs simulated truth, n = 1,821): whole-sequence exact match **98.1 %**, line-1 regimen correct **98.7 %**,
line start within 7 d **99.8 %**. The remaining errors are the two intrinsically ambiguous cases built into the data — late-added agents and long
toxicity pauses. The 5 × 5 window × gap sensitivity grid ([figure](results/figures/08_lot_sensitivity.png)) shows that no setting is perfect and
that very wide windows (≥ 70 d) are worse.

**Overall survival by first-line regimen** ([KM](results/figures/02_km_os_regimen.png), log-rank p = 2e-10)

| First line | n | Median OS, mo (95 % CI) | 24-mo OS |
|---|---|---|---|
| EGFR TKI | 124 | 21.4 (16.6–24.7) | 43 % |
| IO + chemo | 433 | 14.9 (13.5–16.3) | 32 % |
| IO mono | 113 | 10.7 (9.2–14.3) | 30 % |
| Platinum chemo | 945 | 10.4 (9.6–11.8) | 22 % |

**IO + chemo vs chemo** (EGFR-negative/unknown, index ≥ Jun-2018, n = 759; simulated true HR = 0.70)

| Method | HR (95 % CI) |
|---|---|
| Unadjusted | 0.70 (0.60–0.82) |
| Multivariable Cox | 0.78 (0.66–0.91) |
| IPTW (max \|SMD\| 0.19 → 0.01) | 0.79 (0.67–0.92) |

All three intervals contain the truth. The single-run multivariable estimate sits ~1.2 SD from truth; the Monte-Carlo study below
shows that this is noise, not bias. The treatment effect satisfies proportional hazards (p = 0.27); the only PH violation is the
"ECOG unknown" indicator, which pools patients of different performance status.

**Monte-Carlo validation** — pipeline re-run on 10 independently generated datasets ([results](results/simulation_study_summary.csv)):

| Estimand | Method | Mean log-HR | Truth | Bias | 95 % CI coverage |
|---|---|---|---|---|---|
| IO + chemo vs chemo | Multivariable Cox | −0.346 | −0.350 | +0.004 | 100 % |
| IO + chemo vs chemo | IPTW (marginal HR) | −0.289 | −0.350 | +0.061 | 90 % |
| IO + chemo vs chemo | Unadjusted | −0.356 | −0.350 | −0.006 | 80 % |
| 2L therapy (true null) | **Naive ever-treated** | **−0.600** | 0 | **−0.600** | **0 %** |
| 2L therapy (true null) | Landmark 6 mo | −0.002 | 0 | −0.002 | 100 % |
| 2L therapy (true null) | Time-varying Cox | −0.022 | 0 | −0.022 | 100 % |

Takeaways: (1) the conditional Cox model is unbiased; (2) IPTW is attenuated — expected, since it targets the *marginal* HR, which is
non-collapsible; (3) confounding by indication is deliberately mild in this simulator, so unadjusted looks fine on average but
under-covers; (4) **immortal-time bias is huge** — a treatment with no effect appears to cut mortality by ~45 % (HR 0.55) unless exposure is
handled as time-varying or by landmark ([figure](results/figures/09_immortal_time_bias.png)).

More figures: [Love plot](results/figures/05_love_plot.png) · [IPTW-weighted KM](results/figures/06_km_iptw.png) ·
[TTNT](results/figures/03_km_ttnt_regimen.png) · [forest plot](results/figures/04_forest_adjusted_cox.png) ·
[swimmer plot of derived lines](results/figures/07_swimmer_lot.png) · [line transitions](results/figures/11_line_transitions.png) ·
[truth recovery](results/figures/10_truth_recovery.png).

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py --regen      # generate data + full analysis -> results/ (a few minutes)
python -m pytest -q                 # 26 tests: LOT rules, QC, cohort integrity, truth recovery, IPTW balance
python simulation_study.py --seeds 10   # Monte-Carlo validation (≈10 min)
```

## Limitations
Synthetic data demonstrate method correctness, not clinical findings. Loss to follow-up is non-informative by construction; ECOG is handled
with a missing-category indicator rather than multiple imputation; progression is not observable (hence TTNT); the LOT rules are rule-based and
parameter-sensitive by nature. See [docs/methods.md](docs/methods.md) §7.

## Possible extensions
Multiple imputation of ECOG/PD-L1 · competing-risk models for TTNT · target-trial-emulation framing with clone-censor-weight ·
OMOP CDM mapping · running the same code on a real, access-controlled oncology dataset (e.g. SEER-Medicare, Flatiron) under the proper data-use agreement.

---

Author: Erick Kiprotich Yegon, epidemiologist and data scientist (real-world evidence, HEOR, causal inference) · Portfolio: https://erickyegon.github.io · LinkedIn: https://linkedin.com/in/erickyegon
