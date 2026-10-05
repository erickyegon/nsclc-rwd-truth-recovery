"""Figures for the README / report. Colour-blind-safe (Okabe-Ito) palette, consistent styling."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

from . import config as C

PAL = {"Platinum chemo": "#7F7F7F", "IO + chemo": "#0072B2", "IO mono": "#009E73", "EGFR TKI": "#D55E00"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / name)
    plt.close(fig)


def attrition(att: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(att))[::-1]
    ax.barh(y, att["n_remaining"], color="#0072B2")
    for yi, (n, ex) in zip(y, zip(att["n_remaining"], att["n_excluded"])):
        ax.text(n + att["n_remaining"].max() * 0.01, yi, f"{n:,}" + (f"  (-{ex:,})" if ex else ""), va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(att["step"], fontsize=8.5)
    ax.set_xlim(0, att["n_remaining"].max() * 1.22)
    ax.set_xlabel("Patients remaining")
    ax.set_title("Cohort attrition")
    ax.grid(axis="y", visible=False)
    _save(fig, "01_attrition.png")


def km_by_group(coh, group, time, event, title, xlabel, fname, order=None, tmax=60):
    fig, ax = plt.subplots(figsize=(8, 5.6))
    fits = []
    for g in order or sorted(coh[group].unique()):
        d = coh[coh[group] == g]
        kmf = KaplanMeierFitter(label=f"{g} (n={len(d)})").fit(d[time], d[event])
        kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12, color=PAL.get(g), show_censors=True,
                                   censor_styles={"ms": 3, "marker": "|"})
        fits.append(kmf)
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend(frameon=False)
    add_at_risk_counts(*fits, ax=ax, rows_to_show=["At risk"])
    _save(fig, fname)


def forest(summary: pd.DataFrame, title: str, fname: str, labels: dict | None = None, xlim=(0.2, 5)):
    s = summary.copy()
    s = s.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(s) + 1.4))
    y = np.arange(len(s))
    ax.errorbar(s["hr"], y, xerr=[s["hr"] - s["hr_low"], s["hr_high"] - s["hr"]], fmt="s", color="#0072B2",
                ecolor="#0072B2", capsize=2, ms=4)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels([(labels or {}).get(i, i) for i in s.index], fontsize=9)
    ax.set_xlabel("Hazard ratio (95% CI), log scale  —  <1 favours lower hazard of death")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _save(fig, fname)


def love_plot(smd_before, smd_after, fname="05_love_plot.png"):
    order = smd_before.abs().sort_values().index
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(order) + 1.4))
    y = np.arange(len(order))
    ax.scatter(smd_before[order].abs(), y, color="#D55E00", label="Unweighted", zorder=3)
    ax.scatter(smd_after[order].abs(), y, color="#0072B2", label="IPT-weighted", zorder=3)
    ax.axvline(0.1, color="k", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xlabel("|Standardised mean difference|")
    ax.set_title("Covariate balance: IO + chemo vs platinum chemo")
    ax.legend(frameon=False)
    _save(fig, fname)


def weighted_km_plot(fits, fname="06_km_iptw.png", tmax=48):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for g, lab, col in [(1, "IO + chemo", PAL["IO + chemo"]), (0, "Platinum chemo", PAL["Platinum chemo"])]:
        fits[g].plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12, color=col, label=lab)
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Months since first-line start")
    ax.set_ylabel("Weighted overall survival")
    ax.set_title("IPTW-adjusted overall survival (EGFR-negative/unknown, index ≥ Jun-2018)")
    ax.legend(frameon=False)
    _save(fig, fname)


def swimmer(lines: pd.DataFrame, coh: pd.DataFrame, n=28, seed=7, fname="07_swimmer_lot.png"):
    from .drugs import regimen_class
    rng = np.random.default_rng(seed)
    pool = coh[(coh["event"] == 1)].index.intersection(lines[lines["line"] >= 2]["patient_id"])
    ids = rng.choice(pool, size=min(n, len(pool)), replace=False)
    sub = coh.loc[ids].sort_values("os_months")
    colors = {**PAL, "Other": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for i, (pid, r) in enumerate(sub.iterrows()):
        for ln in lines[lines["patient_id"] == pid].itertuples():
            s = (ln.start - r["index_date"]).days / C.DAYS_PER_MONTH
            e = max((ln.end - r["index_date"]).days / C.DAYS_PER_MONTH, s + 0.5)
            ax.barh(i, e - s, left=s, color=colors[regimen_class(ln.regimen)], height=0.62, edgecolor="white", lw=0.6)
            ax.text(s + (e - s) / 2, i, str(ln.line), ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
        ax.plot(r["os_months"], i, marker="X", color="k", ms=6)
    ax.set_yticks([])
    ax.set_xlabel("Months since first-line start")
    ax.set_title("Derived lines of therapy for 28 random decedents with ≥2 lines\n(number in bar = line; colour = regimen class; X = death)", fontsize=10)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], labels=list(colors),
              frameon=False, loc="lower right", title="Regimen class")
    ax.grid(axis="y", visible=False)
    _save(fig, fname)


def lot_sensitivity(grid: pd.DataFrame, fname="08_lot_sensitivity.png"):
    pv = grid.pivot(index="window_days", columns="gap_days", values="sequence_exact_match")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(pv.values, cmap="Blues", vmin=pv.values.min() - 0.01, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pv.columns)), pv.columns)
    ax.set_yticks(range(len(pv.index)), pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center",
                    color="white" if pv.values[i, j] > pv.values.mean() else "black", fontsize=8.5)
    ax.set_xlabel("Treatment-free gap that opens a new line (days)")
    ax.set_ylabel("Regimen window (days)")
    ax.set_title("LOT accuracy vs simulated truth\n(patient-level exact regimen sequence)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, fname)


def immortal(df: pd.DataFrame, fname="09_immortal_time_bias.png"):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(df))[::-1]
    cols = ["#D55E00", "#E69F00", "#0072B2"]
    for yi, (_, r), c in zip(y, df.iterrows(), cols):
        ax.errorbar(r["hr"], yi, xerr=[[r["hr"] - r["hr_low"]], [r["hr_high"] - r["hr"]]], fmt="o", color=c, capsize=3, ms=8)
    ax.axvline(1, color="k", ls="--", lw=0.9)
    ax.text(1.02, y.max() + 0.45, "true HR = 1.0 (no effect)", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"])
    ax.set_xlabel("HR for death, second-line therapy vs none (95% CI)")
    ax.set_title("Immortal-time bias: a null treatment looks protective unless handled")
    ax.grid(axis="y", visible=False)
    _save(fig, fname)


def truth_recovery(tab: pd.DataFrame, fname="10_truth_recovery.png"):
    t = tab.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(t) + 1.4))
    y = np.arange(len(t))
    ax.errorbar(t["est_log_hr"], y, xerr=[t["est_log_hr"] - t["lo"], t["hi"] - t["est_log_hr"]], fmt="o", color="#0072B2", capsize=2, label="Estimate (95% CI)")
    ax.scatter(t["true_log_hr"], y, marker="D", color="#D55E00", zorder=4, label="Simulated truth")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index, fontsize=9)
    ax.set_xlabel("log hazard ratio")
    ax.set_title("Does the pipeline recover the simulated effects?")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.grid(axis="y", visible=False)
    _save(fig, fname)


def transitions(lines: pd.DataFrame, cohort_ids, fname="11_line_transitions.png"):
    from .drugs import regimen_class
    l = lines[lines["patient_id"].isin(cohort_ids)].copy()
    l["cls"] = l["regimen"].map(regimen_class)
    l1 = l[l["line"] == 1].set_index("patient_id")["cls"]
    l2 = l[l["line"] == 2].set_index("patient_id")["cls"]
    tab = pd.crosstab(l1, l2.reindex(l1.index).fillna("No 2L (death/censor)"))
    tab = tab.div(tab.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    im = ax.imshow(tab.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=20, ha="right")
    ax.set_yticks(range(tab.shape[0]), tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            ax.text(j, i, f"{tab.values[i, j]:.0%}", ha="center", va="center",
                    color="white" if tab.values[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Second line")
    ax.set_ylabel("First line")
    ax.set_title("Treatment-pattern transitions (row %)")
    ax.grid(False)
    _save(fig, fname)
    return tab
