"""Data QC, cohort definition with attrition, and baseline covariate / outcome derivation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .drugs import regimen_class
from .lot import build_admin_table, derive_lines

CODE_NSCLC, CODE_SCLC = "254637007", "254632001"
CODE_STAGE = {"424132000": 1, "425048006": 2, "422968005": 3, "423121009": 4}
CODE_PRIOR_CANCER = {"254837009", "93761005"}
CODE_COMORB = {"13645005", "44054006", "709044004", "84114007"}
CODE_BRAIN = "94225005"
HIST = {"SYN-HIST-AD": "Adenocarcinoma", "SYN-HIST-SQ": "Squamous", "SYN-HIST-NOS": "Other/NOS"}


def load_raw(raw_dir: Path = C.RAW_DIR) -> dict[str, pd.DataFrame]:
    parse = {"patients": ["BIRTHDATE", "DEATHDATE"], "conditions": ["START", "STOP"], "medications": ["START", "STOP"],
             "observations": ["DATE"], "encounters": ["START", "STOP"], "procedures": ["START", "STOP"]}
    return {k: pd.read_csv(Path(raw_dir) / f"{k}.csv", parse_dates=v, dtype={"CODE": str, "VALUE": str})
            for k, v in parse.items()}


def run_qc(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Flag and fix data-quality problems; return cleaned tables and a QC log."""
    t = {k: v.copy() for k, v in raw.items()}
    log = []

    n0 = len(t["medications"])
    t["medications"] = t["medications"].drop_duplicates()
    log.append(("Exact duplicate medication rows", n0 - len(t["medications"]), "dropped"))

    death = t["patients"].set_index("Id")["DEATHDATE"]
    e = t["encounters"]
    post = e["START"] > e["PATIENT"].map(death)
    log.append(("Encounters dated after recorded death", int(post.sum()), "dropped"))
    t["encounters"] = e[~post]

    m = t["medications"]
    birth = m["PATIENT"].map(t["patients"].set_index("Id")["BIRTHDATE"])
    log.append(("Medication start before birth", int((m["START"] < birth).sum()), "flagged"))

    inverted = t["medications"]["STOP"] < t["medications"]["START"]
    log.append(("Medication STOP before START", int(inverted.sum()), "flagged"))

    dx = (t["conditions"][t["conditions"]["CODE"] == CODE_NSCLC].groupby("PATIENT")["START"].min())
    adm = build_admin_table(t["medications"])
    first = adm.groupby("patient_id")["start"].min()
    pre = (first < first.index.map(dx)).sum()
    log.append(("Patients with antineoplastic before NSCLC diagnosis (mis-dated)", int(pre), "excluded in cohort step"))
    return t, pd.DataFrame(log, columns=["check", "n_flagged", "action"])


def _first_start(df, code_set_or_code, key="CODE"):
    mask = df[key].isin(code_set_or_code) if isinstance(code_set_or_code, (set, list)) else df[key] == code_set_or_code
    return df[mask].groupby("PATIENT")["START"].min()


def build_cohort(t: dict[str, pd.DataFrame], lines: pd.DataFrame, cutoff=C.DATA_CUTOFF):
    pat = t["patients"].rename(columns={"Id": "patient_id"}).set_index("patient_id")
    cond, obs, enc = t["conditions"], t["observations"], t["encounters"]

    dx = _first_start(cond, CODE_NSCLC).rename("dx_date")
    df = pat.join(dx)
    sclc = set(cond.loc[cond["CODE"] == CODE_SCLC, "PATIENT"])
    stage_rows = cond[cond["CODE"].isin(CODE_STAGE)].sort_values("START").groupby("PATIENT").last()
    df["stage"] = stage_rows["CODE"].map(CODE_STAGE)
    prior = cond[cond["CODE"].isin(CODE_PRIOR_CANCER)].groupby("PATIENT")["START"].min()
    l1 = lines[lines["line"] == 1].set_index("patient_id")
    df["index_date"] = l1["start"]
    df["line1_regimen"] = l1["regimen"]
    df["line1_start_to_dx"] = (df["index_date"] - df["dx_date"]).dt.days
    df["age"] = (df["dx_date"] - df["BIRTHDATE"]).dt.days / 365.25

    steps = []
    keep = pd.Series(True, index=df.index)

    def step(name, cond_mask):
        nonlocal keep
        keep = keep & cond_mask.fillna(False)
        steps.append((name, int(keep.sum())))

    steps.append(("All patients in EHR extract", int(len(df))))
    step("NSCLC diagnosis (SNOMED 254637007), no SCLC", df["dx_date"].notna() & ~df.index.isin(sclc))
    step("Age >= 18 at diagnosis", df["age"] >= 18)
    step("Advanced disease (stage III-IV) at diagnosis", df["stage"].isin([3, 4]))
    prior_dt = df.index.to_series().map(prior)
    step("No prior other primary cancer (breast/colorectal)", ~(prior_dt < df["dx_date"]))
    step("Received systemic anticancer therapy (>=1 derived line)", df["index_date"].notna())
    step("Therapy not dated before diagnosis (date integrity)", df["line1_start_to_dx"] >= 0)
    step(f"First-line start within {C.MAX_DX_TO_INDEX_DAYS} d of diagnosis", df["line1_start_to_dx"] <= C.MAX_DX_TO_INDEX_DAYS)

    # outcomes: death, else censor at last recorded activity (capped at data cutoff)
    last_enc = enc.groupby("PATIENT")["START"].max()
    last_med = t["medications"].groupby("PATIENT")["START"].max()
    df["last_activity"] = pd.concat([last_enc, last_med], axis=1).max(axis=1).reindex(df.index).clip(upper=pd.Timestamp(cutoff))
    df["event"] = df["DEATHDATE"].notna().astype(int)
    df["end_date"] = df["DEATHDATE"].where(df["event"] == 1, df["last_activity"])
    df["os_months"] = (df["end_date"] - df["index_date"]).dt.days / C.DAYS_PER_MONTH
    step("At least 1 day of follow-up after index", df["os_months"] > 0)

    coh = df[keep].copy()
    attrition = pd.DataFrame(steps, columns=["step", "n_remaining"])
    attrition["n_excluded"] = -attrition["n_remaining"].diff().fillna(0).astype(int)

    # ---- baseline covariates relative to index
    idx = coh["index_date"]
    o = obs.merge(idx.rename("index_date"), left_on="PATIENT", right_index=True)
    o["rel"] = (o["DATE"] - o["index_date"]).dt.days

    def latest(code, lo, hi):
        sub = o[(o["CODE"] == code) & (o["rel"] >= lo) & (o["rel"] <= hi)].sort_values("DATE")
        return sub.groupby("PATIENT")["VALUE"].last()

    ecog = pd.to_numeric(latest("89247-1", -90, 7), errors="coerce")
    coh["ecog"] = ecog.reindex(coh.index).map(lambda v: "Unknown" if pd.isna(v) else "3+" if v >= 3 else str(int(v)))
    pd1 = pd.to_numeric(latest("SYN-PDL1", -90, 14), errors="coerce").reindex(coh.index)
    coh["pdl1"] = pd.cut(pd1, [-1, 0, 49, 100], labels=["<1%", "1-49%", ">=50%"]).astype(object).fillna("Unknown")
    coh["egfr"] = latest("SYN-EGFR", -90, 14).reindex(coh.index).fillna("Unknown")
    smoke_map = {"Never smoker": "Never", "Former smoker": "Former", "Current every day smoker": "Current"}
    coh["smoking"] = latest("72166-2", -400, 60).reindex(coh.index).map(smoke_map).fillna("Unknown")
    c = cond.merge(idx.rename("index_date"), left_on="PATIENT", right_index=True)
    coh["brain_mets"] = coh.index.isin(c.loc[(c["CODE"] == CODE_BRAIN) & (c["START"] <= c["index_date"] + pd.Timedelta(days=30)), "PATIENT"]).astype(int)
    com = c[c["CODE"].isin(CODE_COMORB) & (c["START"] < c["index_date"])].groupby("PATIENT")["CODE"].nunique()
    coh["n_comorbid"] = com.reindex(coh.index).fillna(0).astype(int)
    hist = c[c["CODE"].isin(HIST)].groupby("PATIENT")["CODE"].first().map(HIST)
    coh["histology"] = hist.reindex(coh.index).fillna("Other/NOS")
    coh["male"] = (coh["GENDER"] == "M").astype(int)
    coh["index_year"] = coh["index_date"].dt.year
    coh["regimen_class"] = coh["line1_regimen"].map(regimen_class)

    # ---- time to next treatment (TTNT): next line start or death
    l2 = lines[lines["line"] == 2].set_index("patient_id")["start"]
    nxt = l2.reindex(coh.index)
    t_death = coh["DEATHDATE"]
    ttnt_end = pd.concat([nxt, t_death], axis=1).min(axis=1)
    coh["ttnt_event"] = ttnt_end.notna().astype(int)
    ttnt_end = ttnt_end.fillna(coh["last_activity"])
    coh["ttnt_months"] = (ttnt_end - coh["index_date"]).dt.days / C.DAYS_PER_MONTH
    coh["second_line_start"] = nxt
    coh["second_line_months"] = (nxt - coh["index_date"]).dt.days / C.DAYS_PER_MONTH
    return coh, attrition
