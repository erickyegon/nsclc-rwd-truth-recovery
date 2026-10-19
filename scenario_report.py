"""Tables T1-T8 and figures F1-F7 for the Phase 1 scenario study (planned in docs/estimands.md section 8).

    python scenario_report.py                 # every cell that has replicate files and a truth file for the current settings
    python scenario_report.py --cells S0 S3_null

Everything is computed from results/phase1/rows_<cell>.csv, diag_<cell>.csv and the hash-named truth files. Output goes to
results/phase1/report/ (CSV and Markdown tables, PNG figures). A cell with fewer replicates than its design value is marked PARTIAL.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import scenario_study as S  # noqa: E402
from oncrwd import metrics  # noqa: E402

OUT = S.OUT / "report"
DESIGN_R = {c: S.CELL_R.get(c, 500) for c in S.CELL_ORDER}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots 1-4 of the reference palette, fixed order = fixed estimator
EST = [("km", "KM (unadjusted)", "#2a78d6"), ("iptw", "IPTW-KM", "#eb6834"), ("gcomp", "G-computation (Cox)", "#1baf7a"), ("aipw", "AIPW", "#eda100")]
CELL_LABEL = {"S0": "S0 baseline", "S3_null": "S3 null (HR 1)", "S3_m070": "S3 log-HR -0.70", "S3_p020": "S3 log-HR +0.20",
              "S1_mod": "S1 confounding, moderate", "S1_strong": "S1 confounding, strong", "S2_mod": "S2 positivity, moderate", "S2_severe": "S2 positivity, severe"}


def fm(v, mc, nd=3, pct=False):
    if pd.isna(v):
        return "n/a"
    if pct:
        return f"{100 * v:.1f} ({100 * mc:.1f})"
    return f"{v:.{nd}f} ({mc:.{nd}f})"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def save_table(name, df, note=""):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text((note + "\n\n" if note else "") + md_table(df) + "\n", encoding="utf8")


def load(cells):
    data = {}
    for c in cells:
        rf, df_ = S.OUT / f"rows_{c}.csv", S.OUT / f"diag_{c}.csv"
        if not (rf.exists() and df_.exists()):
            continue
        try:
            th = S.load_truth(c)
        except SystemExit:
            continue
        rows, diag = pd.read_csv(rf), pd.read_csv(df_)
        data[c] = {"rows": rows, "diag": diag, "truth": th, "summary": metrics.summarise(rows, th), "R": diag.seed.nunique(),
                   "partial": diag.seed.nunique() < DESIGN_R[c]}
    return data


def style(ax, title, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def legend_est(ax, loc="best"):
    for key, label, col in EST:
        ax.plot([], [], "o", color=col, label=label, markersize=7)
    ax.legend(frameon=False, fontsize=8, loc=loc)


def dot_figure(data, metric, mcse, xlabel, title, fname, ref=None, band=None, estimand="rmst24", zero=False):
    cells = list(data)
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(cells) * 4 / 3 + 2.0), facecolor=SURFACE)
    for i, c in enumerate(cells):
        s = data[c]["summary"].set_index("method")
        for j, (key, label, col) in enumerate(EST):
            m = f"{estimand}_{key}"
            if m not in s.index or pd.isna(s.loc[m].get(metric)):
                continue
            y = i + (j - 1.5) * 0.17
            v, e = s.loc[m, metric], 1.96 * s.loc[m, mcse]
            ax.errorbar(v, y, xerr=e, fmt="o", color=col, ecolor=col, elinewidth=1.4, capsize=2, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([CELL_LABEL[c] + (" (partial)" if data[c]["partial"] else "") for c in cells], fontsize=8.5, color=INK2)
    ax.invert_yaxis()
    if ref is not None:
        ax.axvline(ref, color=INK2, linewidth=1, linestyle="--")
    if band:
        ax.axvspan(band[0], band[1], color=GRID, alpha=0.6, linewidth=0)
    style(ax, title, xlabel)
    legend_est(ax, "lower right")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(cells):
    OUT.mkdir(parents=True, exist_ok=True)
    data = load(cells)
    if not data:
        raise SystemExit("no cell has rows, diag and a truth file for the current settings")
    # ---- T1 truth
    t1 = pd.DataFrame([{"cell": c, "population": d["truth"]["n_population"], "E2 24-mo diff": fm(d["truth"]["rmst_diff_24"], d["truth"]["rmst_diff_24_mcse"]),
                        "E2 12-mo diff": fm(d["truth"]["rmst_diff_12"], d["truth"]["rmst_diff_12_mcse"]),
                        "E3 surv diff 12 / 24": f"{d['truth']['surv_diff_12']:.3f} / {d['truth']['surv_diff_24']:.3f}",
                        "marginal log-HR": f"{d['truth']['marginal_log_hr']:.3f}", "conditional log-HR": f"{d['truth']['conditional_log_hr']:.2f}",
                        "analytic 24-mo check": f"{d['truth']['rmst_diff_24_analytic']:.3f}", "hash": d["truth"]["provenance"]["hash"]} for c, d in data.items()])
    save_table("T1_truth", t1, "T1. Truth by cell; Monte Carlo SE in brackets.")
    # ---- T2 at risk
    rows2 = []
    for c, d in data.items():
        a = metrics.atrisk_summary(d["diag"]).set_index("quantity")
        rec = {"cell": c + (" (partial)" if d["partial"] else ""), "R": d["R"]}
        for q in ("risk24_io", "risk24_chemo", "risk12_io", "risk12_chemo"):
            rec[q.replace("risk", "at risk ").replace("_", " ") + " median [p10, min]"] = f"{a.loc[q, 'median']:.0f} [{a.loc[q, 'p10']:.0f}, {a.loc[q, 'min']:.0f}]"
            rec[q.replace("risk", "n<20 ").replace("_", " ")] = int(a.loc[q, "replicates_below_20"])
        rows2.append(rec)
    save_table("T2_at_risk", pd.DataFrame(rows2), "T2. Number at risk at 12 and 24 months per arm: median [10th percentile, minimum] and replicates with fewer than 20.")
    # ---- T3/T4/T5
    def perf(prefixes, name, note):
        recs = []
        for c, d in data.items():
            for _, r in d["summary"].iterrows():
                if not any(r.method.startswith(p) for p in prefixes):
                    continue
                recs.append({"cell": c + (" (partial)" if d["partial"] else ""), "estimator": r.method, "R": int(r.R), "truth": f"{r.truth:.3f}",
                             "bias (MCSE)": fm(r.get("bias"), r.get("bias_mcse")), "emp. SE (MCSE)": fm(r.get("emp_se"), r.get("emp_se_mcse")),
                             "model-SE ratio (MCSE)": fm(r.get("model_se_ratio"), r.get("model_se_ratio_mcse"), 2),
                             "coverage % (MCSE)": fm(r.get("coverage"), r.get("coverage_mcse"), pct=True),
                             "rejection % (MCSE)": fm(r.get("rejection"), r.get("rejection_mcse"), pct=True),
                             "MSE (MCSE)": fm(r.get("mse"), r.get("mse_mcse")), "median CI width": f"{r.get('median_ci_width', float('nan')):.3f}",
                             "failure %": f"{100 * r.fail_rate:.1f}"})
        save_table(name, pd.DataFrame(recs), note)
    perf(["rmst24"], "T3_primary_rmst24", "T3. PRIMARY: difference in 24-month RMST (months). Bias, empirical SE, model-SE ratio, coverage, rejection rate, MSE with Monte Carlo SE in brackets.")
    perf(["rmst12", "surv"], "T4_secondary_rmst12_surv", "T4. Secondary: 12-month RMST difference and survival-probability differences.")
    perf(["cox"], "T5_hazard_ratio_estimators", "T5. Secondary: Cox estimators on the log-HR scale (multivariable Cox vs the conditional truth; unadjusted and IPTW Cox vs the marginal truth).")
    # ---- T6 S1 calibration
    if "S0" in data:
        recs = []
        base = data["S0"]["rows"].query("method == 'cox_unadjusted'")["est"].dropna()
        for c, tgt in (("S1_mod", -0.10), ("S1_strong", -0.25)):
            if c in data:
                x = data[c]["rows"].query("method == 'cox_unadjusted'")["est"].dropna()
                diff = x.mean() - base.mean()
                mc = np.sqrt(x.var(ddof=1) / len(x) + base.var(ddof=1) / len(base))
                recs.append({"cell": c, "target bias": tgt, "achieved bias (MCSE)": fm(diff, mc), "replicates": len(x)})
        if recs:
            save_table("T6_S1_calibration_check", pd.DataFrame(recs), "T6. Unadjusted log-HR minus its mean in S0, against the calibration targets.")
    # ---- T7 positivity diagnostics
    recs = [{"cell": c + (" (partial)" if d["partial"] else ""), "median min PS": f"{d['diag']['ps_min'].median():.3f}",
             "median share outside [0.05, 0.95]": f"{100 * d['diag']['ps_share_lt05'].median():.1f}%"} for c, d in data.items() if c in ("S0", "S2_mod", "S2_severe")]
    if recs:
        save_table("T7_positivity_diagnostics", pd.DataFrame(recs), "T7. Propensity-score diagnostics per replicate, median over replicates.")
    # ---- T8 truth precision
    recs = []
    for c, d in data.items():
        for _, r in d["summary"].iterrows():
            if pd.notna(r.get("emp_se")) and r.truth_mcse:
                ratio = r.truth_mcse / r.emp_se
                recs.append({"cell": c, "estimator": r.method, "truth MCSE / empirical SE": f"{ratio:.3f}", "flag (>= 0.10)": "FLAG" if ratio >= 0.10 else ""})
    save_table("T8_truth_precision", pd.DataFrame(recs), "T8. Truth Monte Carlo SE divided by the empirical SE of each estimator (rule: below 0.10).")

    # ---- figures
    # F1 at risk at 24 months
    fig, ax = plt.subplots(figsize=(8.2, 3.6), facecolor=SURFACE)
    pos = 0
    ticks, labels = [], []
    for c, d in data.items():
        for k, (col, arm, colr) in enumerate((("risk24_io", "IO + chemo", "#2a78d6"), ("risk24_chemo", "Platinum chemo", "#eb6834"))):
            bp = ax.boxplot(d["diag"][col], positions=[pos + k * 0.35], widths=0.28, vert=True, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set(facecolor=colr, edgecolor=SURFACE, alpha=0.9)
            for m in bp["medians"]:
                m.set(color=INK, linewidth=1.3)
        ticks.append(pos + 0.17)
        labels.append(c.replace("_", "\n"))
        pos += 1
    ax.axhline(20, color=INK2, linestyle="--", linewidth=1)
    ax.text(pos - 0.5, 21, "20 at risk", fontsize=8, color=INK2, ha="right")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("number at risk at 24 months", fontsize=9, color=INK2)
    style(ax, "F1. Number at risk at 24 months per arm (blue IO + chemo, orange platinum chemo)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "F1_at_risk.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    dot_figure(data, "bias", "bias_mcse", "bias of the 24-month RMST difference (months), 95% band = 1.96 x Monte Carlo SE", "F2. Bias", "F2_bias.png", ref=0.0)
    dot_figure(data, "coverage", "coverage_mcse", "95% interval coverage (grey band = 93% to 97%)", "F3. Coverage (failures counted as misses)", "F3_coverage.png", ref=0.95, band=(0.93, 0.97))
    dot_figure(data, "model_se_ratio", "model_se_ratio_mcse", "average model SE / empirical SE (grey band = 0.9 to 1.1)", "F4. Model-SE ratio", "F4_model_se_ratio.png", ref=1.0, band=(0.9, 1.1))
    # F5 null cell
    if "S3_null" in data:
        s = data["S3_null"]["summary"].set_index("method")
        fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=SURFACE)
        for j, (key, label, col) in enumerate(EST):
            m = f"rmst24_{key}"
            if m in s.index:
                ax.errorbar(j, 100 * s.loc[m, "rejection"], yerr=196 * s.loc[m, "rejection_mcse"], fmt="o", color=col, ecolor=col, capsize=3, markersize=7, markeredgecolor=SURFACE)
        ax.axhline(5, color=INK2, linestyle="--", linewidth=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels([e[1] for e in EST], fontsize=8)
        ax.set_ylabel("rejection rate at the null (%)", fontsize=9, color=INK2)
        style(ax, "F5. False-positive rate when the true effect is zero (dashed line = 5%)")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(OUT / "F5_null_rejection.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)
    # F6 distributions
    show = [c for c in ("S0", "S1_strong", "S2_severe") if c in data]
    if show:
        fig, axes = plt.subplots(1, len(show), figsize=(4.3 * len(show), 3.8), facecolor=SURFACE, sharey=False)
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, show):
            r = data[c]["rows"]
            for j, (key, label, col) in enumerate(EST):
                x = r[(r.method == f"rmst24_{key}") & (r.failure.isna() | (r.failure == ""))]["est"].dropna()
                ax.scatter(np.random.default_rng(j).normal(j, 0.07, len(x)), x, s=7, color=col, alpha=0.35, linewidths=0)
                ax.plot([j - 0.25, j + 0.25], [x.mean()] * 2, color=INK, linewidth=1.6)
            ax.axhline(data[c]["truth"]["rmst_diff_24"], color=INK2, linestyle="--", linewidth=1)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["KM", "IPTW", "G-comp", "AIPW"], fontsize=8)
            style(ax, CELL_LABEL[c])
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("24-month RMST difference (months)", fontsize=9, color=INK2)
        fig.suptitle("F6. Estimates by replicate (black bar = mean, dashed = truth)", x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout()
        fig.savefig(OUT / "F6_estimate_distributions.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)
    # F7 positivity
    pc = [c for c in ("S0", "S2_mod", "S2_severe") if c in data]
    if len(pc) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), facecolor=SURFACE)
        for ax, (metric, mcse, ttl, ref) in zip(axes, (("coverage", "coverage_mcse", "coverage of IPTW-KM", 0.95), ("model_se_ratio", "model_se_ratio_mcse", "model-SE ratio of IPTW-KM", 1.0))):
            for c in pc:
                s = data[c]["summary"].set_index("method").loc["rmst24_iptw"]
                x = 100 * data[c]["diag"]["ps_share_lt05"].median()
                ax.errorbar(x, s[metric], yerr=1.96 * s[mcse], fmt="o", color="#eb6834", ecolor="#eb6834", capsize=3, markersize=7, markeredgecolor=SURFACE)
                ax.annotate(CELL_LABEL[c].replace("S2 positivity, ", "").replace("S0 baseline", "baseline"), (x, s[metric]), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK2)
            ax.axhline(ref, color=INK2, linestyle="--", linewidth=1)
            ax.set_xlabel("median % of patients with PS outside [0.05, 0.95]", fontsize=9, color=INK2)
            style(ax, ttl)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
        fig.suptitle("F7. Positivity stress", x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout()
        fig.savefig(OUT / "F7_positivity.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)
    status = {c: {"replicates": d["R"], "design": DESIGN_R[c], "partial": bool(d["partial"])} for c, d in data.items()}
    (OUT / "report_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=S.CELL_ORDER)
    main(ap.parse_args().cells)
