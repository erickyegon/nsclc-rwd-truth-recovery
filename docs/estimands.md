# Estimands, truth and pre-specified performance metrics (Phase 1)

Everything on this page was fixed **before** any main-study replicate was run. A change after results are seen is added to the
"Deviations" list at the bottom with its date and reason. All data are synthetic: this is a methods check, not a clinical finding.

## 1. Target trial (IO + platinum chemotherapy vs platinum chemotherapy)

| Component | Specification |
|---|---|
| Eligibility | Stage III/IV NSCLC; first-line therapy within 90 days of diagnosis; index on or after 2018-06-01; EGFR not positive (negative or untested); no prior cancer; at least 1 day of follow-up |
| Strategies | (A) IO + platinum chemotherapy at first line; (B) platinum chemotherapy alone |
| Assignment | Observational emulation; in the generator, conditional on ECOG, PD-L1 and era (plus, in scenario S1, the prognostic score) |
| Time zero | First-line start (eligibility, assignment and start of follow-up coincide; no immortal time for this contrast) |
| Outcome | Overall survival; follow-up to death, loss to follow-up or 2024-12-31 |
| Assumptions | Exchangeability given measured covariates, positivity, consistency, non-informative censoring (each stressed in a scenario) |

## 2. Estimands

| ID | Estimand | Role |
|---|---|---|
| **E2** | **Difference in 24-month restricted mean survival time, IO+chemo minus chemo, marginal over the eligible population (months)** | **Primary, in every scenario at every sample size** |
| E2-12 | The same at 12 months | Secondary, in every scenario |
| E3 | Survival-probability difference at 12 and 24 months | Secondary |
| E1 | Conditional log hazard ratio given measured covariates | Secondary (truth = the configured effect, log-HR -0.35 at baseline) |
| E4 | Marginal log hazard ratio (treatment independent of covariates in a pseudo-population, no censoring) | Descriptive; target of the unadjusted and IPTW Cox estimators |

**At-risk condition.** In every cell the median, 10th percentile and minimum number at risk at 12 and 24 months in each arm, and the
number of replicates with fewer than 20 at risk in an arm, are reported. No horizon is changed after seeing results; instability of the
24-month estimate in small samples is reported as a finding.

## 3. Truth

Computed in `src/oncrwd/truth.py`, which imports no estimator, cohort or line-of-therapy code and does not call the generator's survival draw.

1. The target population is the set of patients that enter the analysis cohort in large generator runs (240,000 simulated patients per cell, 40 chunks of 6,000), with their **true** covariates.
2. For each patient both potential outcomes are drawn 20 times with `scipy.stats.weibull_min` from a separate PCG64 stream, using the survival model stated in `config.py` (Weibull proportional hazards, shape 1.1, median 700 days at linear predictor 0). Truth values are means of min(T, tau) differences, survival-indicator differences, and a Cox fit on the stacked pseudo-population.
3. Cross-check: the same RMST difference by numerical integration of the closed-form survival function. A test fails when the two differ by more than 4 Monte Carlo SE.
4. **Truth precision rule:** the truth's Monte Carlo SE must be below 10% of the empirical SE of the estimator being judged (`truth_precision_ok` in every summary).

## 4. Estimators and the truth each is judged against

| Estimator | Targets |
|---|---|
| Multivariable Cox | E1 |
| Unadjusted Cox, IPTW Cox (stabilised, 1/99 trimmed) | E4 |
| KM RMST (unadjusted), IPTW-KM RMST, g-computation (Cox outcome model), AIPW (IPCW, Cox outcome model, logistic propensity) at 24 and 12 months | E2 and E2-12 |
| KM survival difference at 12 and 24 months | E3 |

IPTW-RMST and g-computation use a 100-resample bootstrap with the propensity / outcome model refitted in every resample and a normal-approximation interval; unweighted KM RMST uses the Greenwood-type variance; AIPW uses the plug-in influence-function variance (nuisance-estimation error ignored, so the model-SE ratio is the check on it).

## 5. Performance metrics (Morris, White and Crowther, Stat Med 2019)

A *failure* is a replicate with no estimate or no finite standard error. Bias, empirical SE, model-SE ratio, MSE and CI width use successful replicates; coverage and rejection counts treat failures as non-coverage and non-rejection and are also reported among successes only. A cell with more than 5% failures is flagged.

| Metric | Definition | Monte Carlo SE |
|---|---|---|
| Bias | mean(est) - truth | sqrt(var(est)/R) |
| Empirical SE | sd(est) | EmpSE / sqrt(2(R-1)) |
| Model-SE ratio | sqrt(mean(SE^2)) / EmpSE | delta method with MCSE(ModSE) = sqrt(var(SE^2) / (4 R ModSE^2)) |
| 95% coverage | proportion of intervals containing the truth | sqrt(p(1-p)/R) |
| MSE | mean((est - truth)^2) | sqrt(sum((e^2 - MSE)^2) / (R(R-1))) |
| Rejection rate | proportion of intervals excluding 0 (type-I error when the truth is 0, power otherwise) | sqrt(p(1-p)/R) |
| Failure rate | failures / R | sqrt(p(1-p)/R) |
| Median CI width | median(hi - lo) | IQR reported |

## 6. Seeds, replicates and scenarios

Burned exploratory seeds (never reused): 101, 102, 201 to 220, 301 to 308, 5001. Pilot/calibration seeds: 7,000,000 and above. Truth populations: 9,000,000 and above. Main replicates: `1001 + 10000 * cell_index + r`.

Priority order and replicates: P1 S0 baseline (R = 1,000, n = 6,000); P2 S3 true effect null / -0.70 / +0.20 (R = 500 each); P3 S1 measured confounding, moderate and strong (R = 500); P4 S2 positivity, moderate and severe (R = 500). Later priorities are in `phase1_truth_recovery/PHASE1_DESIGN_v2.md` and are not part of this first run.

## 7. Calibration of scenario levels (pilot seeds, before the main runs)

Pilot seeds 7,000,000+, 40 chunks of 6,000 patients per setting, pooled. 'Unadjusted bias' = unadjusted Cox log-HR minus its value at gamma = 0 (more negative = IO looks better than it is).

**S1 confounding (gamma on the standardised prognostic score; prognostic score mean 0.851, SD 0.480):**

| gamma | unadjusted log-HR | unadjusted bias |
|---|---|---|
| 0.0 | -0.354 | +0.000 |
| 0.5 | -0.517 | -0.163 |
| 1.0 | -0.638 | -0.284 |
| 1.5 | -0.752 | -0.398 |
| 2.0 | -0.808 | -0.454 |
| 3.0 | -0.879 | -0.525 |

Chosen by linear interpolation: **moderate gamma = 0.307** (target bias -0.10), **strong gamma = 0.861** (target bias -0.25). The bias actually achieved in the main runs is reported with the S1 results (interpolation is approximate).

**S2 positivity (multiplier on the ECOG and PD-L1 terms of the assignment logit; propensity from the fitted logistic model in the analysis cohort):**

| multiplier | min PS | 1st percentile of PS | share with PS outside [0.05, 0.95] |
|---|---|---|---|
| 1.0 | 0.1 | 0.270 | 0.0% |
| 2.0 | 0.01 | 0.091 | 0.3% |
| 3.0 | 0.0035 | 0.027 | 3.7% |
| 4.0 | 1.5e-06 | 0.004 | 11.9% |
| 5.0 | 4.9e-07 | 0.001 | 24.2% |
| 6.0 | 1e-07 | 0.001 | 31.5% |

Rule fixed in advance: moderate = smallest multiplier whose 1st-percentile PS is below 0.10 (**2.0**); severe = smallest multiplier with at least 10% of patients outside [0.05, 0.95] (**4.0**).

## 8. Planned results: tables, figures and reading rules (written before any P1-P4 result was read)

All tables are produced by one script from the replicate-level files `results/phase1/rows_<cell>.csv`, `diag_<cell>.csv` and the hash-named truth files. Nothing is typed by hand. Every table carries the Monte Carlo SE of each number in brackets.

**Tables**

| # | Table | Rows x columns | Source |
|---|---|---|---|
| T1 | Truth by cell | cell x (E2 24-month, E2 12-month, E3 12/24-month, marginal log-HR, conditional log-HR, truth Monte Carlo SE, analytic cross-check, population size, hash) | `truth_<cell>_<hash>.json` |
| T2 | Number at risk | cell x arm x {12, 24 months}: median, 10th percentile, minimum, replicates below 20 | `diag_<cell>.csv` |
| T3 | **Primary: E2 at 24 months** | cell x estimator (KM, IPTW-KM, g-computation, AIPW): bias, empirical SE, model-SE ratio, 95% coverage, rejection rate, MSE, median CI width, failure rate | `summary_<cell>.csv` |
| T4 | Secondary: E2 at 12 months and E3 | same columns, same estimators plus KM survival difference | `summary_<cell>.csv` |
| T5 | Secondary: hazard-ratio estimators | cell x {multivariable Cox vs E1, unadjusted and IPTW Cox vs E4}: same columns | `summary_<cell>.csv` |
| T6 | S1 calibration check | strong and moderate cells: unadjusted log-HR bias achieved in the main runs vs the -0.10 / -0.25 targets (mean over replicates, with Monte Carlo SE) | `rows_S1_*.csv`, `rows_S0.csv` |
| T7 | S2 positivity diagnostics | cell: median minimum PS and median share of patients with PS outside [0.05, 0.95] (the minimum PS and the share are what `diag_<cell>.csv` records) | `diag_<cell>.csv` |
| T8 | Truth-precision check | cell x estimator: truth Monte Carlo SE divided by empirical SE; any value at or above 0.10 is flagged | `summary_<cell>.csv` |

**Figures** (static PNG, one estimator colour each, axes labelled in months for E2; the same estimator order in every panel)

| # | Figure | Content |
|---|---|---|
| F1 | At-risk by arm | Distribution of the number at risk at 24 months per arm across replicates, with the 20-patient line |
| F2 | Bias | Dot plot of bias for each estimator and cell with +/-1.96 Monte Carlo SE bars; zero line |
| F3 | Coverage | Coverage per estimator and cell with +/-1.96 Monte Carlo SE bars against the nominal 95% |
| F4 | Model-SE ratio | Ratio with Monte Carlo SE bars; reference line at 1 |
| F5 | Null cells | Rejection rate in the null cells (S3 null) against 5%, with Monte Carlo SE bars |
| F6 | Distribution of estimates | Violin / strip plot of the 24-month RMST estimates per estimator against the truth line, S0 and the strongest stress cell of each scenario |
| F7 | Positivity | S2: model-SE ratio and coverage of IPTW-KM against the share of patients outside [0.05, 0.95] |

**Reading rules (reporting conventions, not hypothesis tests; no cell is called a pass or a fail on one metric)**

- *Bias* is described as "consistent with zero" when |bias| < 2 x its Monte Carlo SE; otherwise the signed value is quoted with its SE. Practical size is judged against the empirical SE: |bias| below one-tenth of the empirical SE is negligible by that yardstick.
- *Coverage* is described as "consistent with nominal" when 95% lies within 2 Monte Carlo SE of the estimate. Otherwise the shortfall or excess is quoted. Coverage among successful replicates is shown beside the failure-inclusive figure.
- *Model-SE ratio* within 0.9 to 1.1, allowing for its Monte Carlo SE, is described as "standard errors about right"; below 0.9 as "standard errors too small", above 1.1 as "too large".
- *Null cell*: the rejection rate is compared with 5% using its Monte Carlo SE.
- *Failures*: any estimator-cell with more than 5% failures is flagged and its other metrics are labelled "among successes".
- *Truth precision*: any estimator-cell where the truth Monte Carlo SE is at least 10% of the empirical SE is flagged, and its bias is not interpreted until the truth population is enlarged.
- Results are reported whichever way they fall, including cases in which a method that the earlier 10-seed table suggested was fine (for example the unadjusted estimator) misses under larger R.
- Questions the P1-P4 results are meant to answer: (1) does each RMST estimator recover E2 at baseline, with honest intervals; (2) does the answer change when assignment is confounded through all measured prognostic variables (S1), when positivity is stressed (S2), and when the true effect is null, larger or harmful (S3); (3) does the earlier IPTW "bias" correspond to the marginal HR truth (E4) rather than the conditional one (E1).

## 9. Deviations and known issues

| Date | Item | Detail |
|---|---|---|
| 2026-10-18 | Stale S0 truth file (fixed) | A 3-chunk smoke-test truth (2,519 patients) was saved as `truth_S0.json` and skipped by the driver's "file exists" check. Caught before any summary used it; rebuilt at 40 chunks. Truth files are now named `truth_<cell>_<hash>.json`, where the hash covers scenario settings (with calibrated values), number and size of chunks, Monte Carlo draws, seed base and the generator parameters; `summarise` refuses a file whose hash does not match, and a test checks that the sample size changes the name. All eight truths were rebuilt under the new names with the same seeds and compared with the earlier 40-chunk files on population size, E2 at 24 and 12 months, E3 at 12 and 24 months, the marginal log-HR and the analytic cross-check: **all eight are identical**, so the truth computation is reproducible. The stale 3-chunk file and the unhashed copies are removed from the repository. |
| 2026-10-18 | Seed disjointness | `seed_registry()` assigns every pilot, truth and main seed from the same functions the code uses; `test_seed_purposes_never_overlap` checks that burned, pilot (LP, confounding, positivity), truth and main seeds are pairwise disjoint for up to 1,000 replicates and 99 chunks per purpose. |
| 2026-10-18 | Known issue: bootstrap stream shared the data seed (checked; P1-P4 stand) | In `replicate()` the bootstrap generator is `default_rng(seed)`, the same integer that seeds the data generator, so the two streams start identically. The bootstrap only resamples cohort rows to obtain standard errors for IPTW-KM and g-computation RMST (24 and 12 months). **Check:** for the first 100 main seeds of cells S0 and S1_strong, the cohort is regenerated and the four bootstrap-based estimators are run twice, once with the original stream and once with an independent stream `default_rng([seed, 1])`. Decision rule, fixed before running: P1-P4 stand if, for every estimator in both cells, (i) the point estimates are identical, (ii) the mean log ratio of standard errors (independent / original) is within 2 Monte Carlo SE of 0, and (iii) the paired difference in 95% coverage is within 2 Monte Carlo SE of 0 (SE from the paired differences). Otherwise the bootstrap step is rerun for P1-P4 with the independent stream. The result is recorded below the table. |
| 2026-10-18 | Warning probe (convergence) | `scenario_study.py` silences all warnings at import, so non-convergence is recorded only when an estimator raises (counted as a failure per replicate and reported as the failure rate per cell). To see whether silent convergence warnings occur, `warning_probe.py` ran the real `replicate()` with all warnings on for 20 replicates in S0 and 20 in S2_severe (the positivity stress cell). Result: the only warning in either cell was the scikit-learn `FutureWarning` about `penalty` (4,100 per cell, 205 per replicate, one per `LogisticRegression` fit including the bootstrap refits). **No convergence warnings and no estimator failures in the 40 probed replicates.** This is a sample, not a census: a rare convergence event in an unprobed replicate would not have been logged. No model call was changed. The software versions that produced the results are in `results/phase1/environment.txt` and `requirements-lock.txt`; `requirements.txt` allows `scikit-learn>=1.9,<1.10`. |

**Result of the bootstrap-stream check (2026-10-19; 100 paired seeds per cell, 4 estimators per cell).** Rule (i): maximum absolute difference in point estimates; rule (ii): mean log ratio of SE (independent / original) with its Monte Carlo SE, rule met when |mean| < 2 x MCSE; rule (iii): paired difference in 95% coverage (independent - original) with its Monte Carlo SE, met when |difference| < 2 x MCSE.

| Cell | Estimator | (i) max est. diff | (ii) mean log SE ratio (MCSE) | (iii) coverage original / independent | coverage diff (MCSE) | rules met |
|---|---|---|---|---|---|---|
| S0 | rmst12_gcomp | 0 | -0.0056 (0.0092) | 0.93 / 0.92 | -0.01 (0.0174) | i, ii, iii |
| S0 | rmst12_iptw | 0 | +0.0012 (0.0083) | 0.95 / 0.95 | +0.00 (0.0142) | i, ii, iii |
| S0 | rmst24_gcomp | 0 | -0.0003 (0.0099) | 0.93 / 0.92 | -0.01 (0.0225) | i, ii, iii |
| S0 | rmst24_iptw | 0 | -0.0076 (0.0105) | 0.96 / 0.95 | -0.01 (0.0174) | i, ii, iii |
| S1_strong | rmst12_gcomp | 0 | +0.0061 (0.0118) | 0.95 / 0.97 | +0.02 (0.0200) | i, ii, iii |
| S1_strong | rmst12_iptw | 0 | -0.0063 (0.0097) | 0.96 / 0.95 | -0.01 (0.0174) | i, ii, iii |
| S1_strong | rmst24_gcomp | 0 | -0.0033 (0.0096) | 0.95 / 0.93 | -0.02 (0.0200) | i, ii, iii |
| S1_strong | rmst24_iptw | 0 | -0.0075 (0.0109) | 0.96 / 0.96 | +0.00 (0.0142) | i, ii, iii |

**All three rules are met for all eight estimator-cell combinations, so P1-P4 stand and the bootstrap step is not rerun.** Files: `results/phase1/bootstrap_check_<cell>.csv` (paired estimates) and `bootstrap_check_<cell>_report.csv`; script `bootstrap_stream_check.py`. P1-P4 keep the original stream in every cell, as run. Cells outside P1-P4 (the deferred S5-S8) will use the independent stream `default_rng([seed, 1])`.

| Date | Item | Detail |
|---|---|---|
| 2026-10-19 | S5-S8 deferred, not dropped | The generator options and estimators for S5 (missing ECOG: MCAR, MAR, MNAR, measurement error; multiple imputation), S6 (non-proportional hazards, non-linear confounding, propensity misspecification), S7/S7b (second-line and first-line immortal time), S8 (informative censoring, IPCW) and the unmeasured-confounder cell S1b are **not implemented and have no results**. Decision (Erick, 2026-10-19): finish P1-P4, the summaries, the report generator, the README update and the n = 1,500 / 3,000 / 12,000 runs first. The design for the deferred cells stays in `phase1_truth_recovery/PHASE1_DESIGN_v2.md`; none of the P1-P4 findings may be described as covering these mechanisms. |
| 2026-10-19 | Run interruption and detached watchdog | The P1-P4 driver stopped at 08:52 on 2026-10-19 (S3_p020 at 350 of 500) with `BrokenPipeError` in the pool workers, i.e. the parent process had gone. The cause is not established (see the PR discussion); the machine had been in Modern Standby overnight. The run is resumable (completed seeds are skipped), so no replicate was lost or repeated. It is now launched by `watchdog_phase1.ps1` (detached, heartbeat every 5 minutes, ALERT lines, automatic restart). |
