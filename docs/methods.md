# Methods / statistical analysis plan

Written the way an RWE protocol would be: question → data → design → estimands → methods → limitations.

## 1. Research question
Among adults with advanced (stage III–IV) non-small-cell lung cancer (NSCLC) who start systemic therapy, how does
overall survival (OS) and time to next treatment (TTNT) differ by first-line regimen, and — in the immunotherapy era —
what is the comparative effectiveness of **PD-1 inhibitor + platinum chemotherapy (IO + chemo)** versus
**platinum chemotherapy alone** in patients without an EGFR mutation?

## 2. Data source
Synthetic EHR in the Synthea CSV layout (`patients`, `conditions`, `medications`, `observations`, `encounters`,
`procedures`), 6,000 patients, produced by `src/oncrwd/generate.py` (seeded, reproducible).
Analysts only ever see `data/raw/`. The simulator's ground truth (`data/truth/`) is used only for validation.

Deliberate real-world defects: duplicated rows, brand/generic/strength variants of drug names, supportive and chronic
medications mixed with anticancer agents, carboplatin ↔ cisplatin swaps, late-added second agents, treatment pauses,
mis-dated records (therapy before diagnosis), encounters after death, loss to follow-up (death not captured),
missing/mis-recorded ECOG, untested PD-L1/EGFR, administrative censoring at 2024-12-31.

## 3. Cohort definition (index = first-line start)
| # | Criterion | Rationale |
|---|---|---|
| 1 | NSCLC diagnosis (SNOMED 254637007), no SCLC | disease definition |
| 2 | Age ≥ 18 at diagnosis | adults |
| 3 | Stage III or IV at diagnosis | candidates for systemic therapy |
| 4 | No prior breast/colorectal primary | avoids attributing prior-cancer therapy to NSCLC |
| 5 | ≥ 1 derived line of anticancer therapy | treated population |
| 6 | First therapy not dated before diagnosis | date integrity |
| 7 | First-line start ≤ 90 days after diagnosis | defines treatment initiation for this disease episode |
| 8 | ≥ 1 day follow-up after index | time at risk |

Comparative-effectiveness sub-cohort: index ≥ 2018-06-01 (IO + chemo available), EGFR not positive, first line is
IO + chemo or platinum chemo. EGFR-unknown patients are kept (testing is incomplete in practice).

## 4. Line-of-therapy algorithm
See the docstring of `src/oncrwd/lot.py` (rules R1–R7). Defaults: 28-day regimen window, 90-day treatment-free gap.
Validated against the simulator's true line structure (patient-level exact match, line-1 regimen accuracy,
start-date agreement) and stress-tested over a 5 × 5 grid of window × gap values.

## 5. Variables
* **Exposure** – first-line regimen class: platinum chemo, IO + chemo, IO monotherapy, EGFR TKI.
* **OS** – index → death (recorded in EHR) or censoring at last recorded activity / data cut-off.
* **TTNT** – index → start of second line **or** death (composite), same censoring.
* **Baseline covariates** (measured on or before index): age, sex, stage, histology, ECOG (latest in −90…+7 d, "unknown"
  category), PD-L1 TPS category, EGFR status, smoking, brain metastases, comorbidity count.

## 6. Statistical methods
1. Kaplan–Meier with Greenwood CIs; median OS with CI; 12/24-month survival; RMST to 24 months; multi-group log-rank.
2. Cox PH, unadjusted and multivariable; Schoenfeld-residual PH test (rank time transform).
3. **IPTW**: unpenalised logistic propensity model on all baseline covariates, stabilised ATE weights trimmed at the
   1st/99th percentile, covariate balance by standardised mean difference (target < 0.1), weighted Cox with robust SEs,
   weighted KM. Note that the marginal HR targeted by IPTW is *non-collapsible* and need not equal the conditional HR.
4. **E-value** (VanderWeele & Ding) for sensitivity to unmeasured confounding.
5. **Immortal-time bias demonstration** – effect of receiving second-line therapy (true HR = 1 by construction) via
   naive ever-treated Cox, 6-month landmark analysis, and a time-varying-exposure Cox model.
6. **Truth recovery** – a Cox model on the full treated cohort whose coefficients are compared with the simulated
   log-HRs; plus a multi-seed Monte-Carlo study (`simulation_study.py`) reporting bias and 95 % CI coverage.

## 7. Known limitations (and how they show up in the results)
* ECOG is recorded with 15 % misclassification and is missing in ~10 % of patients; an "unknown" indicator mixes
  hazards and visibly violates proportional hazards (`ph_test_adjusted.csv`). Multiple imputation would be the next step.
* Censoring at last recorded activity assumes non-informative loss to follow-up (true in the simulator, not in reality).
* No progression/response data: real-world progression is not captured, hence TTNT as the pragmatic endpoint.
* Late-added second agents and long toxicity pauses are intrinsically ambiguous for rule-based LOT; the sensitivity
  grid shows the cost of each parameter choice. There is no parameter setting that is perfect.
* Synthetic data demonstrate method correctness, **not** clinical findings. No result here should be cited as evidence
  about real patients.
