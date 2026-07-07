"""Paper figures for the ACP-centred UQ benchmark.

Reads the CSVs produced by eval_acp_benchmark.py / analyze_arms_multi.py and
renders five publication figures to results/acp_benchmark/figures/. Site labels
are romanised so no CJK font is required (the old pipeline emitted DejaVu glyph
warnings for Hangul).

Figures:
  Fig1_coverage_by_method.png   raw vs split vs ACI marginal PICP (target 0.90)
  Fig2_worst_site_recovery.png  per-site PICP by method (worst-site recovery)
  Fig3_reliability.png          reliability curve of the raw Gaussian UQ (pooled)
  Fig4_tail_coverage.png        bloom/tail PICP by method (early-warning slice)
  Fig5_coverage_width.png       coverage-width trade-off (gamma sweep + methods)

Usage: python scripts/make_paper_figures.py
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.uq_scores import interval_ece
from src.visualization import paperstyle as ps
ps.apply()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = f"{BASE}/results/acp_benchmark"
FIGD = f"{OUTD}/figures"
TARGET = 0.90
SITE = {0: "Gongju", 1: "Daecheong", 2: "Gapcheon", 3: "Buyeo", 4: "Yongdam"}
MCOL = ps.PALETTE       # shared palette: raw=grey, split=blue, aci=red
MLAB = ps.LABEL         # shared labels (consistent with F6/F7)
ORDER = ["raw", "split", "aci"]


def _pooled(df, key, method):
    return df[df.Method == method][key].to_numpy(float)


def fig1_coverage(master):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for i, mth in enumerate(ORDER):
        v = _pooled(master, "PICP", mth)
        ax.bar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)) * 1.96,
               color=MCOL[mth], width=0.62, capsize=4)
        ax.text(i, v.mean() + 0.004, f"{v.mean():.3f}", ha="center", fontsize=9)
    ax.axhline(TARGET, ls="--", c="k", lw=1, label=f"target {TARGET:.2f}")
    ax.set_xticks(range(3)); ax.set_xticklabels([MLAB[m] for m in ORDER], rotation=12, fontsize=8)
    ax.set_ylabel("PICP (marginal coverage)"); ax.set_ylim(0.85, 0.98)
    ax.set_title("Marginal coverage by calibration method")
    ax.legend(fontsize=8); fig.tight_layout()
    ps.save(fig, f"{FIGD}/Fig1_coverage_by_method")


def fig2_worst_site(per_site):
    g = per_site.groupby(["Method", "Station_ID"])["PICP"].mean().reset_index()
    sites = sorted(g.Station_ID.unique())
    x = np.arange(len(sites)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for j, mth in enumerate(ORDER):
        vals = [g[(g.Method == mth) & (g.Station_ID == s)]["PICP"].mean() for s in sites]
        ax.bar(x + (j - 1) * w, vals, w, color=MCOL[mth], label=MLAB[mth])
    ax.axhline(TARGET, ls="--", c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([SITE.get(s, str(s)) for s in sites], fontsize=8)
    ax.set_ylabel("PICP per site"); ax.set_ylim(0.5, 1.0)
    ax.set_title("Per-site coverage by calibration method")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.01, "Raw over-covers (≈0.96 marginal); fair comparison is split vs ACI.",
             ha="center", fontsize=7, style="italic", color="#555555")
    ps.save(fig, f"{FIGD}/Fig2_worst_site_recovery")


def fig3_reliability():
    # pool raw (mu, sd, y) across all test predictions -> overall reliability
    mus, sds, ys = [], [], []
    for p in glob.glob(f"{BASE}/results/ablation_reps/r*/*/predictions.csv"):
        d = pd.read_csv(p)
        mus.append(d["Predicted_Mean"].to_numpy(float))
        sds.append(d["Predicted_Std"].to_numpy(float))
        ys.append(d["Actual"].to_numpy(float))
    mu, sd, y = np.concatenate(mus), np.concatenate(sds), np.concatenate(ys)
    levels = np.arange(0.1, 0.96, 0.05)
    ece, per = interval_ece(y, mu, sd, levels)
    nominal = [c for c, _ in per]; emp = [e for _, e in per]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", c="k", lw=1, label="perfect")
    ax.plot(nominal, emp, "-o", c=MCOL["raw"], ms=4, label=f"raw UQ (ECE={ece:.3f})")
    ax.set_xlabel("nominal central coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability of the raw Gaussian UQ (pooled)")
    ax.legend(fontsize=8); ax.set_aspect("equal"); fig.tight_layout()
    ps.save(fig, f"{FIGD}/Fig3_reliability")


def fig4_tail(master):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for i, mth in enumerate(ORDER):
        v = _pooled(master, "tail_PICP", mth)
        ax.bar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)) * 1.96,
               color=MCOL[mth], width=0.62, capsize=4)
        ax.text(i, v.mean() + 0.004, f"{v.mean():.3f}", ha="center", fontsize=9)
    ax.axhline(TARGET, ls="--", c="k", lw=1, label=f"target {TARGET:.2f}")
    ax.set_xticks(range(3)); ax.set_xticklabels([MLAB[m] for m in ORDER], rotation=12, fontsize=8)
    ax.set_ylabel("PICP on bloom slice (top-decile chl-a)")
    ax.set_ylim(0.6, 1.0); ax.set_title("Tail (bloom) coverage — the early-warning slice")
    ax.legend(fontsize=8); fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.01, "Raw over-covers (misses 0.90 target, ~30% wider); compare split vs ACI.",
             ha="center", fontsize=7, style="italic", color="#555555")
    ps.save(fig, f"{FIGD}/Fig4_tail_coverage")


def fig5_tradeoff(master, gamma_csv):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    # method reference points (pooled)
    for mth in ORDER:
        x = _pooled(master, "MPIW", mth).mean()
        yv = _pooled(master, "PICP", mth).mean()
        ax.scatter(x, yv, s=90, color=MCOL[mth], zorder=3, label=MLAB[mth])
    # gamma sweep curve for ACI
    if os.path.exists(gamma_csv):
        gs = pd.read_csv(gamma_csv).sort_values("gamma")
        ax.plot(gs.MPIW_mean, gs.PICP_mean, "-o", c=MCOL["aci"], ms=4, alpha=0.7)
        for _, r in gs.iterrows():
            ax.annotate(f"γ={r.gamma:g}", (r.MPIW_mean, r.PICP_mean),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(TARGET, ls="--", c="k", lw=1)
    ax.set_xlabel("MPIW (interval width, mg/m³)"); ax.set_ylabel("PICP")
    ax.set_title("Coverage–width trade-off")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    ps.save(fig, f"{FIGD}/Fig5_coverage_width")


def main():
    os.makedirs(FIGD, exist_ok=True)
    master = pd.read_csv(f"{OUTD}/master_metrics.csv")
    per_site = pd.read_csv(f"{OUTD}/per_site.csv")
    fig1_coverage(master)
    fig2_worst_site(per_site)
    fig3_reliability()
    fig4_tail(master)
    fig5_tradeoff(master, f"{OUTD}/gamma_sensitivity.csv")
    print("Saved figures ->", FIGD)
    for f in sorted(glob.glob(f"{FIGD}/*.png")):
        print("  ", os.path.basename(f))


if __name__ == "__main__":
    main()
