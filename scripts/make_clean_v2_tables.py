#!/usr/bin/env python3
"""Generate manuscript tables from the clean_reanalysis_v2 calibration outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results" / "clean_reanalysis_v2" / "2026-08-18_clean-v2-sealed2025" / "calibration"
OUT = BASE.parent / "paper" / "latex" / "sections"
ORDER = ["raw", "scp", "ecp", "rcp", "aci"]
LABELS = {
    "raw": "Raw Gaussian",
    "scp": "Static CP",
    "ecp": "Expanding CP",
    "rcp": "Rolling CP",
    "aci": "ACI",
}


def write(name: str, content: str) -> None:
    (OUT / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def calibration_table() -> None:
    data = pd.read_csv(RESULTS / "method_summary.csv").set_index("Method").loc[ORDER]
    rows = []
    for method, row in data.iterrows():
        rows.append(
            f"{LABELS[method]} & {row.PICP:.3f} & {row.Coverage_Gap:.3f} & "
            f"{row.MPIW:.2f} & {row.Winkler:.2f} & {row.Worst_Site_PICP:.3f} & "
            f"{row.Bloom_PICP:.3f} & {row.Bloom_Winkler:.2f} \\\\"
        )
    write(
        "tab_redesign_calibration.tex",
        r"""\begin{table*}[t]
\centering
\caption{Prediction-interval performance on the sealed 2025 evaluation stream, averaged over 75 balanced prediction runs across ten stations in the Geum and Nakdong basins. Coverage targets 0.90; lower absolute gap, width, and Winkler score are better. High-Chl-a outcomes are observations above station-specific 2024 90th-percentile thresholds.}
\label{tab:redesign_calibration}
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrrrrr}
\toprule
Method & PICP & $|$gap$|$ & MPIW & Winkler & \shortstack{Worst-site\\PICP} & \shortstack{High-Chl-a\\PICP} & \shortstack{High-Chl-a\\Winkler} \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table*}""",
    )


def paired_table() -> None:
    data = pd.read_csv(RESULTS / "paired_cell_differences.csv")
    score_rows = []
    event_rows = []
    for comparator in ("scp", "ecp", "rcp"):
        group = data[data["Comparison"] == f"aci_minus_{comparator}"]

        def summary(column: str, digits: int = 2) -> str:
            values = group[column]
            return (
                f"{values.mean():+.{digits}f} "
                f"[{values.min():+.{digits}f},{values.max():+.{digits}f}]"
            )

        wins = int((group["Delta_Winkler"] < 0).sum())
        score_rows.append(
            f"ACI $-$ {LABELS[comparator]} & {summary('Delta_Winkler')} & "
            f"{summary('Delta_Coverage_Gap', 3)} & {wins}/15 \\\\"
        )
        event_rows.append(
            f"ACI $-$ {LABELS[comparator]} & {summary('Delta_MPIW')} & "
            f"{summary('Delta_Bloom_PICP', 3)} \\\\"
        )
    write(
        "tab_redesign_paired.tex",
        r"""\begin{table*}[t]
\centering
\caption{Paired ACI differences across 15 backbone--loss cells after averaging five replicates per cell. Entries are mean [minimum, maximum]. Negative differences favor ACI for Winkler score, coverage gap, and width; positive differences favor ACI for high-Chl-a coverage.}
\label{tab:redesign_paired}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrr}
\toprule
Comparison & $\Delta$Winkler & $\Delta|$gap$|$ & Wins \\
\midrule
"""
        + "\n".join(score_rows)
        + r"""
\bottomrule
\end{tabular}

\vspace{0.6em}

\begin{tabular}{lrr}
\toprule
Comparison & $\Delta$width & $\Delta$high-Chl-a PICP \\
\midrule
"""
        + "\n".join(event_rows)
        + r"""
\bottomrule
\end{tabular}
\end{table*}""",
    )


def point_table() -> None:
    data = pd.read_csv(RESULTS / "point_forecast_summary.csv")
    rows = [f"{row.Model} & {int(row.Runs)} & {row.RMSE:.3f} & {row.MAE:.3f} \\\\" for row in data.itertuples()]
    write(
        "tab_redesign_point.tex",
        r"""\begin{table}[t]
\centering
\caption{Point-forecast performance on the sealed 2025 test year. Deep-ensemble values average the indicated prediction runs; persistence is a single deterministic baseline that repeats the previous observed value.}
\label{tab:redesign_point}
\begin{tabular}{lrrr}
\toprule
Model & Runs & RMSE & MAE \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )


def decision_table() -> None:
    data = pd.read_csv(RESULTS / "decision_summary.csv")
    data = data[data["Threshold_Quantile"].round(2) == 0.90].set_index("Method").loc[ORDER]
    rows = []
    for method, row in data.iterrows():
        rows.append(
            f"{LABELS[method]} & {row.Miss_Rate:.3f} & {row.False_Alarm_Rate:.3f} & "
            f"{row.Cost_Ratio_1:.3f} & {row.Cost_Ratio_5:.3f} & {row.Cost_Ratio_10:.3f} \\\\"
        )
    write(
        "tab_redesign_decision.tex",
        r"""\begin{table}[t]
\centering
\caption{Retrospective alert performance at each station's 2024 90th-percentile Chl-a threshold. Cost columns weight a missed event by $\rho$ relative to a false alarm.}
\label{tab:redesign_decision}
\begin{tabular}{lrrrrr}
\toprule
Method & Miss & False alarm & $\rho=1$ & $\rho=5$ & $\rho=10$ \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )


def selection_table() -> None:
    selected = json.loads((RESULTS / "selected_hyperparameters.json").read_text(encoding="utf-8"))
    data = pd.read_csv(RESULTS / "validation_selection.csv")
    rows = []
    for row in data.sort_values(["Method", "Candidate"]).itertuples():
        chosen = (
            row.Method == "aci" and float(row.Candidate) == float(selected["Selected_gamma"])
        ) or (
            row.Method == "rcp" and int(row.Candidate) == int(selected["Selected_rolling_window"])
        )
        candidate = f"{row.Candidate:.3f}" if row.Method == "aci" else f"{int(row.Candidate)}"
        rows.append(
            f"{LABELS[row.Method]} & {candidate} & {row.PICP:.3f} & {row.Coverage_Gap:.3f} & "
            f"{row.MPIW:.2f} & {row.Winkler:.2f} & {'yes' if chosen else ''} \\\\"
        )
    write(
        "tab_redesign_selection.tex",
        r"""\begin{table}[t]
\centering
\caption{Calibration-only hyperparameter selection on the latter half of each station's 2024 validation stream (per-station counts vary with observation availability). Selection minimizes the mean Winkler score and is frozen before any 2025 outcome is loaded.}
\label{tab:redesign_selection}
\begin{tabular}{lrrrrr l}
\toprule
Method & Candidate & PICP & $|$gap$|$ & MPIW & Winkler & Selected \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    calibration_table()
    paired_table()
    point_table()
    decision_table()
    selection_table()
    print(f"Wrote manuscript tables to {OUT}")


if __name__ == "__main__":
    main()
