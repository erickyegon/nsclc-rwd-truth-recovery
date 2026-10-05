"""Rule-based line-of-therapy (LOT) derivation, plus validation against simulated truth.

Rules (all parameters configurable; defaults in config.py)
  R1  Only recognised NSCLC antineoplastics count; supportive care and chronic meds are ignored.
  R2  Carboplatin and cisplatin are one pharmacologic unit ("platinum"): swapping them is NOT a new line.
  R3  The first antineoplastic administration opens line 1.
  R4  Agents first given within `window` days of the line start form the (combination) regimen.
  R5  A *new* agent after the window starts a new line  (add-on / switch  => progression-driven change).
  R6  Re-starting an agent already in the line after a treatment-free gap > `gap` days starts a new line.
  R7  Dropping an agent from a combination (e.g. carboplatin stops, pembrolizumab + pemetrexed
      continue as maintenance) is NOT a new line.
Known limitations (quantified in the sensitivity analysis): clinically planned but late-added agents
are split off by R5, and long toxicity pauses are split by R6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .drugs import lot_agent, normalise_drug


def build_admin_table(meds: pd.DataFrame) -> pd.DataFrame:
    """Medication rows -> one row per (patient, agent, administration)."""
    m = meds.copy()
    m["drug"] = m["DESCRIPTION"].map(normalise_drug)
    m = m[m["drug"].notna()].copy()
    m["agent"] = m["drug"].map(lot_agent)
    m["start"] = pd.to_datetime(m["START"])
    m["end"] = pd.to_datetime(m["STOP"]).fillna(m["start"])
    m["end"] = m[["start", "end"]].max(axis=1)
    out = (m.rename(columns={"PATIENT": "patient_id"})[["patient_id", "agent", "start", "end"]]
           .drop_duplicates().sort_values(["patient_id", "start", "agent"]).reset_index(drop=True))
    return out


def derive_lines(admin: pd.DataFrame, window: int = C.LOT_WINDOW_DAYS, gap: int = C.LOT_GAP_DAYS) -> pd.DataFrame:
    rows = []
    for pid, g in admin.groupby("patient_id", sort=False):
        cur = None
        lines = []
        for agent, s, e in zip(g["agent"], g["start"], g["end"]):
            if cur is not None:
                gap_d = (s - cur["last_end"]).days
                if agent in cur["agents"]:
                    new_line = gap_d > gap                                    # R6
                else:
                    new_line = not ((s - cur["start"]).days <= window and gap_d <= gap)   # R4 / R5
                if not new_line:
                    cur["agents"].add(agent)
                    if (s - cur["start"]).days <= window:
                        cur["initial"].add(agent)
                    cur["last_end"] = max(cur["last_end"], e)
                    cur["n_admin"] += 1
                    continue
                lines.append(cur)
            cur = {"start": s, "last_end": e, "agents": {agent}, "initial": {agent}, "n_admin": 1}
        if cur is not None:
            lines.append(cur)
        for i, ln in enumerate(lines, 1):
            rows.append((pid, i, "+".join(sorted(ln["initial"])), "+".join(sorted(ln["agents"])),
                         ln["start"], ln["last_end"], ln["n_admin"]))
    return pd.DataFrame(rows, columns=["patient_id", "line", "regimen", "regimen_all", "start", "end", "n_admin"])


def evaluate_against_truth(pred: pd.DataFrame, truth: pd.DataFrame, valid_patients) -> dict:
    """Compare derived LOT with the simulation's true lines (excluding deliberately corrupted records)."""
    pred = pred[pred["patient_id"].isin(valid_patients)]
    truth = truth[truth["patient_id"].isin(valid_patients)]

    def seq(df):
        return df.sort_values(["patient_id", "line"]).groupby("patient_id")["regimen"].agg(tuple)

    ps, ts = seq(pred), seq(truth)
    allp = ps.index.union(ts.index)
    ps, ts = ps.reindex(allp), ts.reindex(allp)
    exact = np.mean([a == b for a, b in zip(ps, ts)])
    n_pred = pred.groupby("patient_id").size().reindex(allp).fillna(0)
    n_true = truth.groupby("patient_id").size().reindex(allp).fillna(0)
    l1p = pred[pred["line"] == 1].set_index("patient_id")["regimen"]
    l1t = truth[truth["line"] == 1].set_index("patient_id")["regimen"]
    both = l1t.index.intersection(l1p.index)
    mg = pred.merge(truth, on=["patient_id", "line"], suffixes=("_p", "_t"))
    return {
        "patients_compared": int(len(allp)),
        "sequence_exact_match": float(exact),
        "n_lines_agreement": float((n_pred == n_true).mean()),
        "line1_regimen_accuracy": float((l1p.reindex(both) == l1t.reindex(both)).mean()),
        "line_start_within_7d": float(((pd.to_datetime(mg["start_p"]) - pd.to_datetime(mg["start_t"])).abs()
                                      <= pd.Timedelta(days=7)).mean()) if len(mg) else float("nan"),
    }


def sensitivity_grid(admin, truth, valid_patients, windows=(14, 28, 42, 56, 70), gaps=(60, 90, 120, 150, 180)):
    out = []
    for w in windows:
        for g in gaps:
            ev = evaluate_against_truth(derive_lines(admin, w, g), truth, valid_patients)
            out.append({"window_days": w, "gap_days": g, **ev})
    return pd.DataFrame(out)
