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

## 8. Deviations

None yet.
