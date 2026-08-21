#!/usr/bin/env python3
"""Create manuscript figures from the leakage-controlled redesign outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results" / "clean_reanalysis_v2" / "2026-08-18_clean-v2-sealed2025" / "calibration"
OUT = BASE.parent / "paper" / "latex" / "figures" / "redesign"
ORDER = ["raw", "scp", "ecp", "rcp", "aci"]
LABELS = {
    "raw": "Raw Gaussian",
    "scp": "Static CP",
    "ecp": "Expanding CP",
    "rcp": "Rolling CP",
    "aci": "ACI",
}
COLORS = {
    "raw": "#8c8c8c",
    "scp": "#4c78a8",
    "ecp": "#72b7b2",
    "rcp": "#f2a541",
    "aci": "#c44e52",
}
HATCHES = {"raw": "", "scp": "//", "ecp": "\\\\", "rcp": "..", "aci": "xx"}
LINESTYLES = {"raw": "-", "scp": "--", "ecp": "-.", "rcp": ":", "aci": "-"}
MARKERS = {"raw": "o", "scp": "s", "ecp": "^", "rcp": "D", "aci": "P"}


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def method_overview():
    summary = pd.read_csv(RESULTS / "method_summary.csv").set_index("Method").loc[ORDER]
    panels = [
        ("PICP", "Marginal coverage", 0.90),
        ("Winkler", "Winkler score (lower is better)", None),
        ("MPIW", "Mean interval width", None),
        ("Bloom_PICP", "Coverage above 2024 site-specific P90", 0.90),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    for panel_idx, (ax, (column, title, target)) in enumerate(zip(axes.flat, panels)):
        values = summary[column].to_numpy()
        bars = ax.bar(
            np.arange(len(ORDER)),
            values,
            color=[COLORS[m] for m in ORDER],
            edgecolor="black",
            linewidth=0.5,
            width=0.72,
        )
        for bar, method in zip(bars, ORDER):
            bar.set_hatch(HATCHES[method])
        ax.set_xticks(np.arange(len(ORDER)), [LABELS[m] for m in ORDER], rotation=20, ha="right")
        ax.set_ylabel(title)
        ax.text(-0.08, 1.04, f"({chr(97 + panel_idx)})", transform=ax.transAxes, fontweight="bold")
        if target is not None:
            ax.axhline(target, color="black", linestyle="--", linewidth=1.1, label="0.90 target")
            ax.set_ylim(min(0.84, values.min() - 0.02), max(0.98, values.max() + 0.02))
            ax.legend(frameon=False, fontsize=8, loc="best")
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3f}" if column.endswith("PICP") else f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)
        sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "Fig1_calibration_comparison")


def site_heatmap():
    site = pd.read_csv(RESULTS / "metrics_by_run_site.csv")
    table = site.groupby(["Site", "Method"])["PICP"].mean().unstack().loc[
        [
            # Geum basin
            "Gongju", "Daecheongho", "Gapcheon", "Buyeo", "Yongdamho",
            # Nakdong basin
            "Seongseo", "Dasan", "Jinju", "Chilseo", "Jeokpo",
        ],
        ORDER,
    ]
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    sns.heatmap(
        table,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        center=0.90,
        vmin=min(0.90, float(table.to_numpy().min())),
        vmax=max(0.90, float(table.to_numpy().max())),
        linewidths=0.7,
        cbar_kws={"label": "2025 empirical coverage"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([LABELS[m] for m in ORDER], rotation=20, ha="right")
    fig.tight_layout()
    save(fig, "Fig2_site_coverage")


def paired_online_comparison():
    cells = pd.read_csv(RESULTS / "method_summary_by_cell.csv")
    metrics = [("Winkler", "Winkler score"), ("Coverage_Gap", "Absolute coverage gap")]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    marker = {"Mamba": "o", "GRU": "s", "iTransformer": "^"}
    for panel_idx, (ax, (metric, title)) in enumerate(zip(axes, metrics)):
        pivot = cells.pivot(index=["Backbone", "Arm"], columns="Method", values=metric).reset_index()
        for backbone, group in pivot.groupby("Backbone"):
            ax.scatter(group["rcp"], group["aci"], s=58, alpha=0.85, marker=marker[backbone], label=backbone)
        low = min(pivot["rcp"].min(), pivot["aci"].min())
        high = max(pivot["rcp"].max(), pivot["aci"].max())
        pad = (high - low) * 0.08 or 0.01
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlabel(f"Rolling CP {title.lower()}")
        ax.set_ylabel(f"ACI {title.lower()}")
        ax.text(-0.10, 1.04, f"({chr(97 + panel_idx)})", transform=ax.transAxes, fontweight="bold")
        sns.despine(ax=ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "Fig3_rolling_vs_adaptive")


def decision_sensitivity():
    data = pd.read_csv(RESULTS / "decision_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharex=True)
    for method in ORDER:
        group = data[data["Method"] == method].sort_values("Threshold_Quantile")
        x = group["Threshold_Quantile"] * 100
        axes[0].plot(
            x,
            group["Miss_Rate"] * 100,
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            color=COLORS[method],
            label=LABELS[method],
        )
        axes[1].plot(
            x,
            group["False_Alarm_Rate"] * 100,
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[0].set_ylabel("Miss rate (%)")
    axes[1].set_ylabel("False-alarm rate (%)")
    for ax in axes:
        ax.set_xlabel("Station-specific 2024 threshold percentile")
        ax.grid(alpha=0.2)
        sns.despine(ax=ax)
    axes[0].text(-0.10, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.10, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "Fig4_decision_sensitivity")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    method_overview()
    site_heatmap()
    paired_online_comparison()
    decision_sensitivity()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
