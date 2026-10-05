"""Synthetic NSCLC oncology EHR generator (Synthea-style CSV schema) with known ground truth.

Why not run Synthea itself?  Synthea's lung-cancer module has no systemic-therapy lines, biomarkers
or realistic treatment selection, so it cannot support a line-of-therapy study.  This generator
writes the same table layout (patients / conditions / medications / observations / encounters /
procedures) but simulates the clinical process explicitly, so that every downstream estimate
can be checked against a known truth.

Design
  * OS (from first-line start) ~ Weibull proportional hazards with the log-HRs in config.TRUE_LOG_HR.
  * Treatment choice is confounded by ECOG, PD-L1, EGFR, histology, age and calendar era.
  * Progression time is generated independently of death given covariates, so receipt of
    second-line therapy has a TRUE null effect (immortal-time-bias demonstration).
  * Real-world mess: duplicates, brand/generic names, supportive meds, cis<->carbo switches,
    delayed second agents, treatment holds, mis-dated records, post-death activity,
    loss to follow-up, missing biomarkers/ECOG, administrative censoring.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .drugs import DESCRIPTION_VARIANTS, SUPPORTIVE_OR_OTHER

TD = pd.Timedelta
SNOMED = {
    "nsclc": ("254637007", "Non-small cell lung cancer (disorder)"),
    "sclc": ("254632001", "Small cell carcinoma of lung (disorder)"),
    "breast": ("254837009", "Malignant tumor of breast (disorder)"),
    "crc": ("93761005", "Primary malignant neoplasm of colon"),
    "stage1": ("424132000", "Non-small cell carcinoma of lung, TNM stage 1 (disorder)"),
    "stage2": ("425048006", "Non-small cell carcinoma of lung, TNM stage 2 (disorder)"),
    "stage3": ("422968005", "Non-small cell carcinoma of lung, TNM stage 3 (disorder)"),
    "stage4": ("423121009", "Non-small cell carcinoma of lung, TNM stage 4 (disorder)"),
    "brain": ("94225005", "Secondary malignant neoplasm of brain (disorder)"),
    "adeno": ("SYN-HIST-AD", "Adenocarcinoma of lung (disorder)"),
    "squam": ("SYN-HIST-SQ", "Squamous cell carcinoma of lung (disorder)"),
    "other": ("SYN-HIST-NOS", "Large cell carcinoma of lung (disorder)"),
}
COMORBIDITIES = {
    "copd": ("13645005", "Chronic obstructive lung disease (disorder)"),
    "t2dm": ("44054006", "Diabetes mellitus type 2 (disorder)"),
    "ckd": ("709044004", "Chronic kidney disease (disorder)"),
    "hf": ("84114007", "Heart failure (disorder)"),
}


def _sig(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _weibull(rng, lp: float, k: float, scale0: float) -> float:
    """Weibull PH draw (days). scale0 = median at lp=0."""
    lam = np.log(2) / scale0 ** k
    return float((-np.log(rng.uniform()) / (lam * np.exp(lp))) ** (1 / k))


def _cycle_dates(rng, start, n_max, limit, interval=21, hold_prob=0.0):
    """Administration dates every `interval` days (+ jitter) strictly before `limit`."""
    dates, d = [], start
    hold_at = int(rng.integers(2, 8)) if rng.random() < hold_prob else -1
    for i in range(n_max):
        if d >= limit:
            break
        dates.append(d)
        extra = int(rng.choice([0, 0, 0, 0, 0, 0, 7, 7, 14]))
        if i == hold_at:
            extra += int(rng.integers(100, 180))      # treatment pause, same line (truth)
        d = d + TD(days=interval + extra)
    return dates, d


def _build_line(rng, drugs, hist, start, limit, plat_pref):
    """Return (admins, init_drugs). admins = [(drug, start, stop)]; drugs may include 'platinum'."""
    drugs = list(drugs)
    admins: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    if {"osimertinib", "erlotinib"} & set(drugs):                       # continuous oral therapy
        d = start
        while d < limit:
            admins.append((drugs[0], d, min(d + TD(days=30), limit)))
            d += TD(days=30)
        return admins, set(drugs)

    if "platinum" in drugs:                                              # induction (+ maintenance)
        partner = "paclitaxel" if "paclitaxel" in drugs else "pemetrexed"
        n_ind = 4 if partner == "paclitaxel" else int(rng.choice([4, 6], p=[0.6, 0.4]))
        has_io = "pembrolizumab" in drugs
        dates, nxt = _cycle_dates(rng, start, n_ind, limit)
        delayed = has_io and len(dates) >= 4 and rng.random() < 0.05     # 2nd agent added late
        switch_at = 2 if rng.random() < 0.08 else 99                     # carbo <-> cis swap
        other = "cisplatin" if plat_pref == "carboplatin" else "carboplatin"
        for j, dt in enumerate(dates):
            admins.append((plat_pref if j < switch_at else other, dt, dt))
            admins.append((partner, dt, dt))
            if has_io and not (delayed and j < 2):
                admins.append(("pembrolizumab", dt, dt))
        if len(dates) == n_ind:
            if has_io:
                maint = ["pembrolizumab"] + (["pemetrexed"] if partner == "pemetrexed" else [])
            else:
                maint = ["pemetrexed"] if (partner == "pemetrexed" and rng.random() < 0.35) else []
            if maint:
                mdates, _ = _cycle_dates(rng, nxt, 35, limit, hold_prob=0.03)
                for dt in mdates:
                    admins += [(m, dt, dt) for m in maint]
        return admins, set(drugs)

    interval = 14 if drugs == ["nivolumab"] else 21                      # single-phase regimens
    n_max = {"pembrolizumab": 35, "nivolumab": 52, "atezolizumab": 35}.get(drugs[0], 6)
    if "ramucirumab" in drugs:
        n_max = 8
    dates, _ = _cycle_dates(rng, start, n_max, limit, interval=interval,
                            hold_prob=0.03 if drugs[0] in {"pembrolizumab", "nivolumab"} else 0.0)
    for dt in dates:
        admins += [(d_, dt, dt) for d_ in drugs]
    return admins, set(drugs)


class _Rows:
    def __init__(self):
        self.patients, self.conds, self.meds, self.obs = [], [], [], []
        self.encs, self.procs, self.truth_lines, self.truth_pat = [], [], [], []


def _gen_nsclc(rng, pid, R: _Rows, cutoff):
    t = C.TRUE_LOG_HR
    dx = pd.Timestamp("2015-01-01") + TD(days=int(rng.integers(0, 3100)))
    age = float(np.clip(rng.normal(68, 9), 35, 92))
    birth = dx - TD(days=int(age * 365.25))
    male = rng.random() < 0.52
    smoke = str(rng.choice(["Never", "Former", "Current"], p=[0.12, 0.55, 0.33]))
    hist = str(rng.choice(["adeno", "squam", "other"], p=[0.62, 0.28, 0.10]))
    stage = int(rng.choice([1, 2, 3, 4], p=[0.18, 0.10, 0.22, 0.50]))
    ecog = int(np.clip(round(rng.normal(0.9 + 0.35 * (stage == 4) + 0.012 * (age - 68), 0.8)), 0, 3))
    n_com = min(4, int(rng.poisson(max(0.2, 1.0 + 0.03 * (age - 68)))))
    coms = list(rng.choice(list(COMORBIDITIES), size=n_com, replace=False)) if n_com else []
    brain = stage == 4 and rng.random() < 0.22
    egfr = hist == "adeno" and rng.random() < {"Never": 0.40, "Former": 0.13, "Current": 0.07}[smoke]
    pdl1_cat = int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15] if egfr else [0.35, 0.35, 0.30]))
    pdl1 = [0, int(rng.integers(1, 50)), int(rng.integers(50, 101))][pdl1_cat]

    lp = (t["age_per_year"] * (age - 68) + t["ecog_per_level"] * ecog + t["stage4_vs_3"] * (stage == 4)
          + t["brain_mets"] * brain + t["comorbidity_per_condition"] * n_com + t["male"] * male
          + t["squamous"] * (hist == "squam") + t["egfr_positive"] * egfr)

    end_loss = dx + TD(days=int(rng.exponential(2800))) if rng.random() < 0.30 else pd.Timestamp.max
    end_obs = min(cutoff, end_loss)

    # --- comorbidity / diagnosis conditions
    def cond(start, code_desc, stop=""):
        R.conds.append((start, stop, pid, code_desc[0], code_desc[1]))
    cond(dx, SNOMED["nsclc"])
    cond(dx, SNOMED[f"stage{stage}"])
    cond(dx, SNOMED[hist])
    for c in coms:
        cond(dx - TD(days=int(rng.integers(365, 3650))), COMORBIDITIES[c])
    if brain:
        cond(dx + TD(days=int(rng.integers(0, 20))), SNOMED["brain"])
    prior_cancer = rng.random() < 0.06
    if prior_cancer:
        cond(dx - TD(days=int(rng.integers(365, 3650))), SNOMED[str(rng.choice(["crc", "breast"]))])

    R.obs.append((dx - TD(days=int(rng.integers(0, 30))), pid, "72166-2", "Tobacco smoking status",
                  {"Never": "Never smoker", "Former": "Former smoker", "Current": "Current every day smoker"}[smoke], ""))

    def obs_biomarkers(ref):
        if rng.random() < 0.75:
            R.obs.append((ref + TD(days=int(rng.integers(5, 30))), pid, "SYN-PDL1",
                          "PD-L1 tumor proportion score", str(pdl1), "%"))
        if rng.random() < (0.80 if hist != "squam" else 0.40):
            R.obs.append((ref + TD(days=int(rng.integers(5, 30))), pid, "SYN-EGFR",
                          "EGFR gene mutation analysis", "Positive" if egfr else "Negative", ""))

    advanced = stage >= 3
    death = pd.NaT
    index = pd.NaT
    regimen_1l = "none"
    lines: list[tuple[int, set, pd.Timestamp, pd.Timestamp]] = []
    admins_all: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    if not advanced:
        death = dx + TD(days=_weibull(rng, lp - 0.8, 1.1, 700))
        if rng.random() < 0.6:
            R.procs.append((dx + TD(days=int(rng.integers(14, 45))), "", pid, "SYN-LOBECT", "Lobectomy of lung"))
        follow_from = dx
    else:
        delay = int(np.clip(rng.lognormal(np.log(30), 0.5), 7, 85))
        if rng.random() < 0.04:
            delay = int(rng.integers(95, 200))                           # late starters -> excluded later
        index = dx + TD(days=delay)
        p_treat = _sig(2.0 - 0.8 * ecog - 0.02 * (age - 68) - 0.2 * n_com)
        treated = rng.random() < p_treat
        era_io = index >= pd.Timestamp(C.IO_ERA_START)
        if treated:
            if egfr and rng.random() < 0.85:
                regimen_1l = "osimertinib" if era_io else "erlotinib"
            elif era_io and rng.random() < _sig(0.9 - 0.5 * ecog + 0.9 * (pdl1 >= 50) - 0.4 * (pdl1 < 1)):
                regimen_1l = "io_mono" if (pdl1 >= 50 and rng.random() < 0.5) else "io_chemo"
            else:
                regimen_1l = "chemo"
            eff = {"osimertinib": t["osimertinib_vs_chemo_egfr"], "erlotinib": t["erlotinib_vs_chemo_egfr"],
                   "io_mono": t["io_mono_vs_chemo"], "io_chemo": t["io_chemo_vs_chemo"], "chemo": 0.0}[regimen_1l]
            lp_t = lp + eff
            T = max(1.0, _weibull(rng, lp_t, 1.1, 700))
            death = index + TD(days=T)
            # --- progression (independent of death given covariates) -> lines of therapy
            prog = index + TD(days=_weibull(rng, lp_t, 1.3, 330))
            progressed = prog < death
            plat = "carboplatin" if rng.random() < 0.75 else "cisplatin"
            partner_nonsq = hist != "squam"
            chemo_drugs = ["platinum", "pemetrexed" if partner_nonsq else "paclitaxel"]
            first = {"osimertinib": ["osimertinib"], "erlotinib": ["erlotinib"], "io_mono": ["pembrolizumab"],
                     "io_chemo": ["pembrolizumab"] + chemo_drugs, "chemo": chemo_drugs}[regimen_1l]
            lim = min(prog if progressed else death, death, end_obs + TD(days=1))
            a1, init1 = _build_line(rng, first, hist, index, lim, plat)
            if a1:
                lines.append((1, init1, a1[0][1], max(a[2] for a in a1)))
                admins_all += a1
                prior = set(first)
                line_no, cur_prog, cur_ok = 1, prog, progressed
                while cur_ok and line_no < 3:
                    p_next = np.clip(0.80 - 0.17 * min(ecog + 1, 3), 0.15, 0.9) * (1.0 if line_no == 1 else 0.55)
                    s_next = cur_prog + TD(days=int(rng.integers(14, 46)))
                    if not (s_next < death and s_next <= end_obs and rng.random() < p_next):
                        break
                    had_io = bool(prior & {"pembrolizumab", "nivolumab", "atezolizumab"})
                    s_era = s_next >= pd.Timestamp(C.IO_ERA_START)
                    if "osimertinib" in prior:
                        nxt = chemo_drugs
                    elif "erlotinib" in prior:
                        nxt = ["osimertinib"] if rng.random() < 0.6 else chemo_drugs
                    elif had_io:
                        nxt = ["docetaxel", "ramucirumab"] if rng.random() < 0.3 else ["docetaxel"]
                        if "docetaxel" in prior:
                            nxt = ["gemcitabine"]
                    elif s_era:
                        nxt = [str(rng.choice(["pembrolizumab", "nivolumab", "atezolizumab"], p=[0.65, 0.30, 0.05]))]
                    else:
                        nxt = ["docetaxel"] if "docetaxel" not in prior else ["gemcitabine"]
                    p_n = s_next + TD(days=_weibull(rng, 0.0, 1.2, 150))
                    ok_n = p_n < death
                    lim_n = min(p_n if ok_n else death, death, end_obs + TD(days=1))
                    a_n, init_n = _build_line(rng, nxt, hist, s_next, lim_n, plat)
                    if not a_n:
                        break
                    line_no += 1
                    lines.append((line_no, init_n, a_n[0][1], max(a[2] for a in a_n)))
                    admins_all += a_n
                    prior |= set(nxt)
                    cur_prog, cur_ok = p_n, ok_n
        else:
            regimen_1l = "none"
            death = dx + TD(days=_weibull(rng, lp + 0.5, 1.1, 700))
        follow_from = index if treated else dx
        obs_biomarkers(dx)
        if rng.random() < 0.9:                                           # ECOG documentation
            rec = ecog + int(rng.choice([-1, 0, 1], p=[0.075, 0.85, 0.075]))
            R.obs.append((index - TD(days=int(rng.integers(3, 28))), pid, "89247-1",
                          "ECOG Performance Status score", str(int(np.clip(rec, 0, 4))), "{score}"))
        if stage == 3 and rng.random() < 0.3:
            R.procs.append((index, "", pid, "SYN-RT", "Radiation therapy"))

    # --- observation window: loss to follow-up / administrative cutoff
    death_recorded = pd.notna(death) and death <= end_obs
    last_event = death if death_recorded else end_obs

    # --- medications (+ supportive / noise), truncated at end_obs
    corrupted = rng.random() < 0.007
    shift = TD(days=int(rng.integers(220, 400))) if corrupted else TD(0)
    chemo_days = set()
    for drug, s, e in admins_all:
        if s > end_obs:
            continue
        desc = str(rng.choice(DESCRIPTION_VARIANTS[drug]))
        R.meds.append((s - shift, (e if e > s else s) - shift, pid, "SYN-" + drug[:6].upper(), desc, "antineoplastic"))
        if e == s:
            chemo_days.add(s)
    for s in sorted(chemo_days):
        if rng.random() < 0.7:
            R.meds.append((s, s, pid, "SYN-SUPP1", "Dexamethasone 4 MG Oral Tablet", "chemotherapy support"))
        if rng.random() < 0.6:
            R.meds.append((s, s, pid, "SYN-SUPP2", "Ondansetron 8 MG Oral Tablet", "chemotherapy support"))
        if rng.random() < 0.3:
            R.meds.append((s + TD(days=1), s + TD(days=1), pid, "SYN-SUPP3", "Pegfilgrastim 6 MG/0.6ML Prefilled Syringe", "chemotherapy support"))
    for desc in rng.choice(SUPPORTIVE_OR_OTHER[3:], size=int(rng.integers(0, 3)), replace=False):
        R.meds.append((dx - TD(days=int(rng.integers(400, 2500))), "", pid, "SYN-CHRON", str(desc), "chronic"))

    # --- encounters: dx visit, treatment visits, sparse follow-up
    enc_dates = {dx}
    enc_dates |= {s for _, s, _ in admins_all if s <= end_obs}
    d = follow_from + TD(days=int(rng.integers(30, 90)))
    while d <= last_event and d <= end_obs:
        enc_dates.add(d)
        d += TD(days=int(rng.integers(60, 120)))
    if pd.notna(death) and death_recorded and rng.random() < 0.003:
        enc_dates.add(death + TD(days=int(rng.integers(5, 40))))          # billing artefact after death
    for ed in sorted(x for x in enc_dates if x <= end_obs + TD(days=40)):
        R.encs.append((ed, ed, pid, "ambulatory", "SYN-ONC", "Oncology encounter"))

    R.patients.append((pid, birth, death if death_recorded else pd.NaT, "M" if male else "F",
                       str(rng.choice(["white", "black", "asian", "other"], p=[0.74, 0.12, 0.08, 0.06])), "NSCLC"))
    for ln, init, s, e in lines:
        if s <= end_obs:
            R.truth_lines.append((pid, ln, "+".join(sorted(init)), s, min(e, end_obs), corrupted))
    R.truth_pat.append(dict(patient_id=pid, age=age, male=male, smoke=smoke, hist=hist, stage=stage, ecog=ecog,
                            n_comorb=n_com, brain=brain, egfr=egfr, pdl1=pdl1, advanced=advanced,
                            regimen_1l_true=regimen_1l, corrupted=corrupted, prior_cancer=prior_cancer,
                            n_lines_true=len([x for x in lines if x[2] <= end_obs]),
                            death_true=death, end_obs=end_obs))


def _gen_other(rng, pid, kind, R, cutoff):
    dx = pd.Timestamp("2015-01-01") + TD(days=int(rng.integers(0, 3500)))
    age = float(np.clip(rng.normal(62, 12), 20, 90))
    birth = dx - TD(days=int(age * 365.25))
    death = dx + TD(days=_weibull(rng, 0.0, 1.0, 2500))
    if kind != "none":
        R.conds.append((dx, "", pid, *SNOMED[kind]))
        if kind == "sclc" and rng.random() < 0.7:                        # SCLC gets platinum-etoposide
            for j in range(4):
                s = dx + TD(days=21 * j + 14)
                if s < death and s <= cutoff:
                    R.meds.append((s, s, pid, "SYN-CARBOP", "Carboplatin 10 MG/ML Injectable Solution", "antineoplastic"))
    else:
        R.conds.append((dx, "", pid, *COMORBIDITIES["t2dm"]))
    R.encs.append((dx, dx, pid, "ambulatory", "SYN-GEN", "General encounter"))
    R.patients.append((pid, birth, death if death <= cutoff else pd.NaT, str(rng.choice(["M", "F"])),
                       str(rng.choice(["white", "black", "asian", "other"])), kind.upper()))


def generate(n: int = C.N_PATIENTS, seed: int = C.SEED, raw_dir: Path = C.RAW_DIR,
             truth_dir: Path = C.TRUTH_DIR) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cutoff = pd.Timestamp(C.DATA_CUTOFF)
    R = _Rows()
    kinds = rng.choice(["nsclc", "sclc", "breast", "crc", "none"], size=n, p=[0.62, 0.07, 0.10, 0.09, 0.12])
    for i, k in enumerate(kinds):
        pid = f"P{i:06d}"
        (_gen_nsclc if k == "nsclc" else lambda r, p, rr, c, _k=k: _gen_other(r, p, _k, rr, c))(rng, pid, R, cutoff)

    f = lambda s: pd.to_datetime(pd.Series(s)).dt.strftime("%Y-%m-%d")      # noqa: E731
    patients = pd.DataFrame(R.patients, columns=["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE", "_kind"]).drop(columns="_kind")
    conds = pd.DataFrame(R.conds, columns=["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"])
    meds = pd.DataFrame(R.meds, columns=["START", "STOP", "PATIENT", "CODE", "DESCRIPTION", "REASONDESCRIPTION"])
    obs = pd.DataFrame(R.obs, columns=["DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE", "UNITS"])
    encs = pd.DataFrame(R.encs, columns=["START", "STOP", "PATIENT", "ENCOUNTERCLASS", "CODE", "DESCRIPTION"])
    procs = pd.DataFrame(R.procs, columns=["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"])

    dup = meds.sample(frac=0.02, random_state=seed)                      # duplicated EHR rows
    meds = pd.concat([meds, dup], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    encs.insert(0, "Id", [f"E{j:07d}" for j in range(len(encs))])

    tables = {"patients": patients, "conditions": conds, "medications": meds, "observations": obs,
              "encounters": encs, "procedures": procs}
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        out = df.copy()
        for c in out.columns:
            if c in {"BIRTHDATE", "DEATHDATE", "START", "STOP", "DATE"}:
                out[c] = pd.to_datetime(out[c].where(out[c] != "", pd.NaT)).dt.strftime("%Y-%m-%d")
        out.to_csv(raw_dir / f"{name}.csv", index=False)
    tl = pd.DataFrame(R.truth_lines, columns=["patient_id", "line", "regimen", "start", "end", "corrupted"])
    tl.to_csv(truth_dir / "truth_lines.csv", index=False)
    pd.DataFrame(R.truth_pat).to_csv(truth_dir / "truth_patients.csv", index=False)
    (truth_dir / "truth_params.json").write_text(json.dumps(C.TRUE_LOG_HR, indent=2))
    return tables


if __name__ == "__main__":
    t = generate()
    print({k: len(v) for k, v in t.items()})
