"""F6 (interval-band time series) and F7 (operational decision) figures.

F6 — qualitative proof that ACES gives sharp + valid intervals through blooms.
     One bloom site (Daecheong, the strongest 2024 bloom, peak ~136 ug/L),
     3 stacked panels raw/split/aci: Actual + predicted mean + shaded interval.
     Static split-conformal fails to contain the bloom peak; online ACI does,
     while staying sharper than raw. Reads the per-step dump (no retraining).

F7 — quantitative decision-layer evaluation under distribution shift (stress).
     (a) miss rate vs alert threshold tau, (b) false-alarm rate vs tau (the
     honest trade-off), (c) rho-cost vs cost ratio. Reads operational_eval.py
     outputs.

Usage: python scripts/make_decision_figures.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualization import paperstyle as ps
ps.apply()
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = f"{BASE}/results/acp_benchmark"
FIGD = f"{OUTD}/figures"
OPD = f"{OUTD}/operational"
SITE = {0: "Gongju", 1: "Daecheong", 2: "Gapcheon", 3: "Buyeo", 4: "Yongdam"}
ORDER = ["raw", "split", "aci"]
ANCHOR_TAU = 52.2          # p90 of pooled Chl-a (alert threshold anchor, ug/L)
F6_CONFIG = "Mamba_Standard_NLL_r0.csv"
F6_SITE = 1                # Daecheong — strongest bloom, clearest split failure


def fig6_bands():
    """3-panel interval bands over 2024 for one bloom site (fair regime)."""
    d = pd.read_csv(f"{OUTD}/perstep/{F6_CONFIG}")
    d = d[(d.kept == True) & (d.Station_ID == F6_SITE)].copy()   # noqa: E712
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date")
    x = d["Date"].to_numpy()
    bands = {"raw": ("raw_lo", "raw_hi"), "split": ("split_lo", "split_hi"),
             "aci": ("aci_lo", "aci_hi")}

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True, sharey=True)
    for ax, mth in zip(axes, ORDER):
        lo, hi = d[bands[mth][0]].to_numpy(), d[bands[mth][1]].to_numpy()
        ax.fill_between(x, lo, hi, color=ps.PALETTE[mth], alpha=0.30,
                        lw=0, label="95% prediction interval")
        ax.plot(x, d["Pred_Mean"], color=ps.PALETTE[mth], lw=1.3, ls="--",
                label="Predicted mean")
        ax.plot(x, d["Actual"], color=ps.ACTUAL_C, lw=1.4, label="Observed Chl-a")
        # how many bloom-tail points the upper bound fails to contain
        thr = np.quantile(d["Actual"], 0.90)
        bl = d["Actual"].to_numpy() >= thr
        miss = float(np.mean(hi[bl] < d["Actual"].to_numpy()[bl]))
        ax.set_title(f"{ps.LABEL[mth]}  —  bloom-tail upper miscoverage {miss:.0%}",
                     loc="left")
        ax.set_ylabel("Chl-a (ug/L)")
        if mth == "raw":
            ax.legend(loc="upper left", ncol=3)
    axes[-1].set_xlabel("2024")
    fig.suptitle(f"Prediction intervals through the 2024 bloom at {SITE[F6_SITE]}",
                 y=0.995, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    ps.save(fig, f"{FIGD}/Fig6_interval_bands")
    print(f"F6 -> {FIGD}/Fig6_interval_bands.pdf/.png")


def fig7_operational():
    """(a) miss vs tau, (b) false-alarm vs tau, (c) rho-cost — stress regime."""
    conf = pd.read_csv(f"{OPD}/confusion_by_tau.csv")
    rho = pd.read_csv(f"{OPD}/rho_cost.csv")
    cs = conf[conf.regime == "stress"]
    rs = rho[rho.regime == "stress"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    # (a) miss rate vs tau
    ax = axes[0]
    for mth in ORDER:
        g = cs[cs.method == mth].sort_values("tau_pct")
        ax.plot(g.tau_pct, 100 * g.miss_rate, marker="o", ms=4,
                color=ps.PALETTE[mth], label=ps.LABEL[mth])
    ax.set_xlabel("Alert threshold tau (percentile of Chl-a)")
    ax.set_ylabel("Miss rate (%)")
    ax.set_title("(a) Missed blooms under shift", loc="left")
    ax.legend()
    # (b) false-alarm vs tau
    ax = axes[1]
    for mth in ORDER:
        g = cs[cs.method == mth].sort_values("tau_pct")
        ax.plot(g.tau_pct, 100 * g.false_alarm_rate, marker="s", ms=4,
                color=ps.PALETTE[mth], label=ps.LABEL[mth])
    ax.set_xlabel("Alert threshold tau (percentile of Chl-a)")
    ax.set_ylabel("False-alarm rate (%)")
    ax.set_title("(b) False alarms (the trade-off)", loc="left")
    # (c) rho-cost
    ax = axes[2]
    for mth in ORDER:
        g = rs[rs.method == mth].sort_values("rho")
        ax.plot(g.rho, g.exp_cost_per_1000, marker="^", ms=4,
                color=ps.PALETTE[mth], label=ps.LABEL[mth])
    ax.set_xscale("log")
    ax.set_xlabel("Cost ratio rho = cost(miss)/cost(false alarm)")
    ax.set_ylabel("Expected cost per 1000 days")
    ax.set_title(f"(c) Cost-weighted error (tau = p90)", loc="left")
    fig.suptitle("Operational early-warning evaluation under distribution shift "
                 "(2024-head calibration)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ps.save(fig, f"{FIGD}/Fig7_operational")
    print(f"F7 -> {FIGD}/Fig7_operational.pdf/.png")


if __name__ == "__main__":
    os.makedirs(FIGD, exist_ok=True)
    fig6_bands()
    fig7_operational()
