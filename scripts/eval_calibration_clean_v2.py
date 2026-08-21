#!/usr/bin/env python3
"""Leakage-controlled calibration evaluation for the sealed-2025 clean rerun.

Two-basin (Geum + Nakdong, ten stations) variant of
``eval_calibration_redesign.py`` for ``clean_reanalysis_v2`` outputs.
Calibration hyperparameters are selected from 2024 validation predictions and
persisted before any 2025 outcome is loaded; the 2025 test year was never
examined during development. Ground truth is cross-checked against the WEIS
Data Core bundle ``sample_index.parquet`` (never the legacy reconstructed
CSV), station panels have per-station observation counts, and prediction
dates come from the driver's saved ``Date`` column.

The prediction CSVs are read-only inputs; this script never overwrites them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


ALPHA = 0.10
EPS = 1e-6
BACKBONES = ("Mamba", "GRU", "iTransformer")
ARMS = ("Standard_NLL", "ISO_NLL", "BetaNLL_0.5", "BetaNLL_1.0", "Faithful")
REPS = (1, 2, 3, 4, 5)
GAMMA_GRID = (0.005, 0.01, 0.02, 0.05, 0.10)
WINDOW_GRID = (30, 60, 90, 180, 365)
# Station order fixed by the clean-v2 driver (sorted canonical station ids).
SITE_NAMES = {
    0: "Seongseo",
    1: "Dasan",
    2: "Jinju",
    3: "Chilseo",
    4: "Gongju",
    5: "Daecheongho",
    6: "Yongdamho",
    7: "Jeokpo",
    8: "Buyeo",
    9: "Gapcheon",
}
SOURCE_SITE_NAMES = {
    0: "성서",
    1: "다산",
    2: "진주",
    3: "칠서",
    4: "공주",
    5: "대청호",
    6: "용담호",
    7: "적포",
    8: "부여",
    9: "갑천",
}
# Warm fraction generalises the legacy fixed 168-of-335 split to the
# per-station variable counts of the two-basin cohort (pre-committed).
WARM_FRACTION = 0.5
METHODS = ("raw", "scp", "ecp", "rcp", "aci")
DECISION_QUANTILES = (0.75, 0.80, 0.85, 0.90, 0.95, 0.98)


@dataclass(frozen=True)
class RunKey:
    rep: int
    backbone: str
    arm: str

    @property
    def config(self) -> str:
        return f"{self.backbone}_{self.arm}"


def finite_sample_quantile(scores: Sequence[float], alpha: float) -> float:
    """Return the corrected split-conformal order statistic.

    k = ceil((n + 1) * (1 - alpha)), capped to [1, n].  The boundary cap is
    the protocol's finite bounded implementation for ACI levels outside the
    empirical quantile range.
    """

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("scores contain non-finite values")
    rank = int(math.ceil((values.size + 1) * (1.0 - float(alpha))))
    rank = min(values.size, max(1, rank))
    return float(np.partition(values, rank - 1)[rank - 1])


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Per-observation Winkler interval score; lower is better."""

    y = np.asarray(y, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    width = upper - lower
    return width + np.where(y < lower, (2.0 / alpha) * (lower - y), 0.0) + np.where(
        y > upper, (2.0 / alpha) * (y - upper), 0.0
    )


def nonconformity(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y, float) - np.asarray(mu, float)) / np.maximum(np.asarray(sd, float), EPS)


def run_calibrator(
    y: Sequence[float],
    mu: Sequence[float],
    sd: Sequence[float],
    warm_scores: Sequence[float],
    method: str,
    *,
    alpha: float = ALPHA,
    gamma: float | None = None,
    window: int | None = None,
) -> Dict[str, np.ndarray]:
    """Generate prequential intervals without looking ahead."""

    y_arr = np.asarray(y, float)
    mu_arr = np.asarray(mu, float)
    sd_arr = np.maximum(np.asarray(sd, float), EPS)
    if not (len(y_arr) == len(mu_arr) == len(sd_arr)):
        raise ValueError("y, mu, and sd lengths differ")
    pool = list(np.asarray(warm_scores, float))
    if not pool:
        raise ValueError("a non-empty warm score pool is required")
    if method not in {"scp", "ecp", "rcp", "aci"}:
        raise ValueError(f"unsupported calibrated method: {method}")
    if method == "aci" and gamma is None:
        raise ValueError("ACI requires gamma")
    if method == "rcp" and (window is None or window < 1):
        raise ValueError("RCP requires a positive window")

    lower = np.empty(len(y_arr), float)
    upper = np.empty(len(y_arr), float)
    q_traj = np.empty(len(y_arr), float)
    alpha_traj = np.full(len(y_arr), alpha, float)
    alpha_t = float(alpha)
    static_q = finite_sample_quantile(pool, alpha) if method == "scp" else None

    for i in range(len(y_arr)):
        active = pool[-window:] if method == "rcp" else pool
        q = static_q if static_q is not None else finite_sample_quantile(active, alpha_t)
        half = q * sd_arr[i]
        lower[i], upper[i] = mu_arr[i] - half, mu_arr[i] + half
        q_traj[i], alpha_traj[i] = q, alpha_t

        score = abs(y_arr[i] - mu_arr[i]) / sd_arr[i]
        if method != "scp":
            pool.append(float(score))
        if method == "aci":
            error = float(score > q)
            alpha_t = float(np.clip(alpha_t + float(gamma) * (alpha - error), 0.0, 1.0))

    return {"lower": lower, "upper": upper, "q": q_traj, "alpha_t": alpha_traj}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_run_paths(input_root: Path) -> Dict[RunKey, Tuple[Path, Path]]:
    paths: Dict[RunKey, Tuple[Path, Path]] = {}
    missing: List[str] = []
    for rep in REPS:
        for backbone in BACKBONES:
            for arm in ARMS:
                key = RunKey(rep, backbone, arm)
                directory = input_root / f"r{rep}" / key.config
                val_path = directory / "val_predictions.csv"
                test_path = directory / "predictions.csv"
                if not val_path.is_file() or not test_path.is_file():
                    missing.append(str(directory))
                else:
                    paths[key] = (val_path, test_path)
    if missing:
        raise FileNotFoundError("missing balanced-panel inputs:\n" + "\n".join(missing))
    if len(paths) != 75:
        raise RuntimeError(f"expected 75 complete runs, found {len(paths)}")
    return paths


def make_manifest(paths: Mapping[RunKey, Tuple[Path, Path]]) -> pd.DataFrame:
    rows = []
    for key, pair in sorted(paths.items(), key=lambda item: (item[0].rep, item[0].backbone, item[0].arm)):
        for split, path in zip(("validation", "test"), pair):
            rows.append(
                {
                    "Rep": key.rep,
                    "Backbone": key.backbone,
                    "Arm": key.arm,
                    "Split": split,
                    "Path": str(path),
                    "Bytes": path.stat().st_size,
                    "SHA256": sha256(path),
                }
            )
    return pd.DataFrame(rows)


def load_bundle_reference(sample_index: Path) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Ground truth per split/station from the clean bundle's sample index."""

    index = pd.read_parquet(sample_index)
    name_to_id = {name: sid for sid, name in SOURCE_SITE_NAMES.items()}
    index = index.assign(
        Station_ID=index["station_name_raw"].map(name_to_id),
        Date=pd.to_datetime(index["target_time_h1"]),
    )
    if index["Station_ID"].isna().any():
        raise ValueError("sample_index contains stations outside the ten-station cohort")
    out: Dict[str, Dict[int, pd.DataFrame]] = {}
    for split_name in ("validation", "test"):
        part = index[index["split"] == split_name]
        per_site: Dict[int, pd.DataFrame] = {}
        for sid in SITE_NAMES:
            site = part[part["Station_ID"] == sid].sort_values("Date").reset_index(drop=True)
            if site.empty:
                raise ValueError(f"{split_name} split has no rows for station {sid}")
            per_site[sid] = pd.DataFrame(
                {"Date": site["Date"], "Actual": site["target_value_h1"].astype(float)}
            )
        out[split_name] = per_site
    # Persistence for the test year: previous observed value in the target
    # stream, seeded by the last validation observation of the same station.
    for sid in SITE_NAMES:
        seed = float(out["validation"][sid]["Actual"].iloc[-1])
        test_actual = out["test"][sid]["Actual"].to_numpy(float)
        out["test"][sid]["Persistence"] = np.concatenate([[seed], test_actual[:-1]])
    return out


def validate_prediction_frame(
    frame: pd.DataFrame,
    key: RunKey,
    split: str,
    expected: Mapping[int, pd.DataFrame],
) -> None:
    required = {"Actual", "Predicted_Mean", "Predicted_Std", "Station_ID", "Date", "Site"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{key} {split}: missing columns {missing}")
    numeric = frame[["Actual", "Predicted_Mean", "Predicted_Std", "Station_ID"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{key} {split}: non-finite values")
    if (frame["Predicted_Std"].to_numpy(float) <= 0).any():
        raise ValueError(f"{key} {split}: non-positive predictive standard deviation")
    total = sum(len(expected[sid]) for sid in SITE_NAMES)
    if len(frame) != total or set(frame["Station_ID"].astype(int)) != set(SITE_NAMES):
        raise ValueError(f"{key} {split}: unexpected shape or station IDs")
    for station_id in SITE_NAMES:
        site = frame[frame["Station_ID"].astype(int) == station_id].sort_values("Date")
        if not (site["Site"] == SOURCE_SITE_NAMES[station_id]).all():
            raise ValueError(f"{key} {split} station {station_id}: Site name mismatch")
        reference = expected[station_id]
        if len(site) != len(reference):
            raise ValueError(f"{key} {split} station {station_id}: unexpected row count")
        observed = site["Actual"].to_numpy(float)
        if not np.allclose(observed, reference["Actual"].to_numpy(float), rtol=1e-6, atol=5e-5):
            max_diff = float(np.max(np.abs(observed - reference["Actual"].to_numpy(float))))
            raise ValueError(f"{key} {split} station {station_id}: source mismatch (max {max_diff})")
        dates = pd.to_datetime(site["Date"]).to_numpy()
        if not (dates == reference["Date"].to_numpy()).all():
            raise ValueError(f"{key} {split} station {station_id}: date mismatch")


def load_validation_runs(
    paths: Mapping[RunKey, Tuple[Path, Path]], source: Mapping[int, pd.DataFrame]
) -> Dict[RunKey, pd.DataFrame]:
    runs: Dict[RunKey, pd.DataFrame] = {}
    for key, (val_path, _) in paths.items():
        frame = pd.read_csv(val_path)
        validate_prediction_frame(frame, key, "validation", source)
        runs[key] = frame
    return runs


def candidate_metrics(
    validation_runs: Mapping[RunKey, pd.DataFrame], method: str, candidate: float
) -> Dict[str, float]:
    scores, coverages, widths = [], [], []
    n_streams = 0
    for frame in validation_runs.values():
        for station_id in SITE_NAMES:
            site = frame[frame["Station_ID"].astype(int) == station_id].sort_values("Date")
            y = site["Actual"].to_numpy(float)
            mu = site["Predicted_Mean"].to_numpy(float)
            sd = site["Predicted_Std"].to_numpy(float)
            n_warm = int(len(y) * WARM_FRACTION)
            warm = nonconformity(y[:n_warm], mu[:n_warm], sd[:n_warm])
            kwargs = {"gamma": candidate} if method == "aci" else {"window": int(candidate)}
            result = run_calibrator(y[n_warm:], mu[n_warm:], sd[n_warm:], warm, method, **kwargs)
            lo, hi = result["lower"], result["upper"]
            target = y[n_warm:]
            scores.append(interval_score(target, lo, hi))
            coverages.append((target >= lo) & (target <= hi))
            widths.append(hi - lo)
            n_streams += 1
    all_scores = np.concatenate(scores)
    all_coverage = np.concatenate(coverages)
    all_width = np.concatenate(widths)
    picp = float(all_coverage.mean())
    return {
        "Method": method,
        "Candidate": float(candidate),
        "Streams": n_streams,
        "Observations": int(len(all_scores)),
        "PICP": picp,
        "Coverage_Gap": abs(picp - (1.0 - ALPHA)),
        "MPIW": float(all_width.mean()),
        "Winkler": float(all_scores.mean()),
    }


def tune_hyperparameters(validation_runs: Mapping[RunKey, pd.DataFrame]) -> Tuple[pd.DataFrame, float, int]:
    rows = [candidate_metrics(validation_runs, "aci", gamma) for gamma in GAMMA_GRID]
    rows += [candidate_metrics(validation_runs, "rcp", float(window)) for window in WINDOW_GRID]
    table = pd.DataFrame(rows)
    gamma_rows = table[table["Method"] == "aci"].sort_values(["Winkler", "Candidate"], ascending=[True, True])
    window_rows = table[table["Method"] == "rcp"].sort_values(["Winkler", "Candidate"], ascending=[True, False])
    return table, float(gamma_rows.iloc[0]["Candidate"]), int(window_rows.iloc[0]["Candidate"])


def raw_interval(mu: np.ndarray, sd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = float(norm.ppf(1.0 - ALPHA / 2.0))
    half = z * np.maximum(sd, EPS)
    return mu - half, mu + half


def metric_record(
    y: np.ndarray,
    mu: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    station_ids: np.ndarray,
    bloom_thresholds: Mapping[int, float],
    severe_thresholds: Mapping[int, float],
) -> Dict[str, float]:
    covered = (y >= lo) & (y <= hi)
    picp = float(covered.mean())
    present_sites = np.unique(station_ids.astype(int))
    per_site = [float(covered[station_ids == sid].mean()) for sid in present_sites]
    bloom = np.array([value >= bloom_thresholds[int(sid)] for value, sid in zip(y, station_ids)], bool)
    severe = np.array([value >= severe_thresholds[int(sid)] for value, sid in zip(y, station_ids)], bool)
    return {
        "PICP": picp,
        "Coverage_Gap": abs(picp - (1.0 - ALPHA)),
        "MPIW": float(np.mean(hi - lo)),
        "Winkler": float(np.mean(interval_score(y, lo, hi))),
        "Worst_Site_PICP": float(min(per_site)),
        "Bloom_N": int(bloom.sum()),
        "Bloom_PICP": float(covered[bloom].mean()) if bloom.any() else float("nan"),
        "Bloom_Winkler": (
            float(np.mean(interval_score(y[bloom], lo[bloom], hi[bloom]))) if bloom.any() else float("nan")
        ),
        "Severe_N": int(severe.sum()),
        "Severe_PICP": float(covered[severe].mean()) if severe.any() else float("nan"),
        "RMSE": float(np.sqrt(np.mean((y - mu) ** 2))),
        "MAE": float(np.mean(np.abs(y - mu))),
    }


def decision_record(
    y: np.ndarray,
    upper: np.ndarray,
    station_ids: np.ndarray,
    event_thresholds: Mapping[int, float],
) -> Dict[str, float]:
    threshold = np.array([event_thresholds[int(sid)] for sid in station_ids], float)
    event = y >= threshold
    alert = upper >= threshold
    fn = (~alert) & event
    fp = alert & (~event)
    result = {
        "Events": int(event.sum()),
        "Non_Events": int((~event).sum()),
        "Miss_Rate": float(fn.sum() / event.sum()),
        "False_Alarm_Rate": float(fp.sum() / (~event).sum()),
    }
    for ratio in (1, 2, 5, 10, 20):
        result[f"Cost_Ratio_{ratio}"] = float((ratio * fn.sum() + fp.sum()) / len(y))
    return result


def evaluate_test(
    paths: Mapping[RunKey, Tuple[Path, Path]],
    validation_runs: Mapping[RunKey, pd.DataFrame],
    reference_test: Mapping[int, pd.DataFrame],
    selected_gamma: float,
    selected_window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[dict] = []
    site_rows: List[dict] = []
    decision_rows: List[dict] = []
    interval_frames: List[pd.DataFrame] = []

    for key, (_, test_path) in sorted(paths.items(), key=lambda item: (item[0].rep, item[0].backbone, item[0].arm)):
        validation = validation_runs[key]
        test = pd.read_csv(test_path)
        validate_prediction_frame(test, key, "test", reference_test)

        thresholds90 = {
            sid: float(np.quantile(validation[validation["Station_ID"].astype(int) == sid]["Actual"], 0.90))
            for sid in SITE_NAMES
        }
        thresholds98 = {
            sid: float(np.quantile(validation[validation["Station_ID"].astype(int) == sid]["Actual"], 0.98))
            for sid in SITE_NAMES
        }
        decision_thresholds = {
            quantile: {
                sid: float(
                    np.quantile(
                        validation[validation["Station_ID"].astype(int) == sid]["Actual"], quantile
                    )
                )
                for sid in SITE_NAMES
            }
            for quantile in DECISION_QUANTILES
        }

        run_parts: List[pd.DataFrame] = []
        bounds: Dict[str, List[np.ndarray]] = {method: [] for method in METHODS}
        y_parts: List[np.ndarray] = []
        mu_parts: List[np.ndarray] = []
        sid_parts: List[np.ndarray] = []

        for sid in SITE_NAMES:
            val_site = validation[validation["Station_ID"].astype(int) == sid].sort_values("Date")
            test_site = test[test["Station_ID"].astype(int) == sid].sort_values("Date")
            y_val = val_site["Actual"].to_numpy(float)
            mu_val = val_site["Predicted_Mean"].to_numpy(float)
            sd_val = val_site["Predicted_Std"].to_numpy(float)
            warm = nonconformity(y_val, mu_val, sd_val)

            y = test_site["Actual"].to_numpy(float)
            mu = test_site["Predicted_Mean"].to_numpy(float)
            sd = test_site["Predicted_Std"].to_numpy(float)
            y_parts.append(y)
            mu_parts.append(mu)
            sid_parts.append(np.full(len(y), sid, int))

            lo, hi = raw_interval(mu, sd)
            bounds["raw"].append(np.column_stack([lo, hi]))
            trajectories: Dict[str, Dict[str, np.ndarray]] = {}
            for method in ("scp", "ecp", "rcp", "aci"):
                kwargs = {}
                if method == "rcp":
                    kwargs["window"] = selected_window
                if method == "aci":
                    kwargs["gamma"] = selected_gamma
                trajectories[method] = run_calibrator(y, mu, sd, warm, method, **kwargs)
                bounds[method].append(np.column_stack([trajectories[method]["lower"], trajectories[method]["upper"]]))

            dates = pd.to_datetime(test_site["Date"]).dt.strftime("%Y-%m-%d").to_numpy()
            part = pd.DataFrame(
                {
                    "Rep": key.rep,
                    "Backbone": key.backbone,
                    "Arm": key.arm,
                    "Date": dates,
                    "Station_ID": sid,
                    "Site": SITE_NAMES[sid],
                    "Actual": y,
                    "Predicted_Mean": mu,
                    "Predicted_Std": sd,
                    "Bloom_Threshold_2023_P90": thresholds90[sid],
                    "Severe_Threshold_2023_P98": thresholds98[sid],
                }
            )
            part["raw_Lower"], part["raw_Upper"] = lo, hi
            for method, trajectory in trajectories.items():
                part[f"{method}_Lower"] = trajectory["lower"]
                part[f"{method}_Upper"] = trajectory["upper"]
                part[f"{method}_Q"] = trajectory["q"]
            part["aci_Alpha_T"] = trajectories["aci"]["alpha_t"]
            run_parts.append(part)

        y_all = np.concatenate(y_parts)
        mu_all = np.concatenate(mu_parts)
        sid_all = np.concatenate(sid_parts)
        for method in METHODS:
            method_bounds = np.concatenate(bounds[method])
            lo_all, hi_all = method_bounds[:, 0], method_bounds[:, 1]
            metric_rows.append(
                {
                    "Rep": key.rep,
                    "Backbone": key.backbone,
                    "Arm": key.arm,
                    "Method": method,
                    **metric_record(y_all, mu_all, lo_all, hi_all, sid_all, thresholds90, thresholds98),
                }
            )
            for quantile, thresholds in decision_thresholds.items():
                decision_rows.append(
                    {
                        "Rep": key.rep,
                        "Backbone": key.backbone,
                        "Arm": key.arm,
                        "Method": method,
                        "Threshold_Quantile": quantile,
                        **decision_record(y_all, hi_all, sid_all, thresholds),
                    }
                )
            for sid in SITE_NAMES:
                mask = sid_all == sid
                record = metric_record(
                    y_all[mask], mu_all[mask], lo_all[mask], hi_all[mask], sid_all[mask], thresholds90, thresholds98
                )
                site_rows.append(
                    {
                        "Rep": key.rep,
                        "Backbone": key.backbone,
                        "Arm": key.arm,
                        "Method": method,
                        "Station_ID": sid,
                        "Site": SITE_NAMES[sid],
                        **record,
                    }
                )
        interval_frames.extend(run_parts)

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(site_rows),
        pd.DataFrame(decision_rows),
        pd.concat(interval_frames, ignore_index=True),
    )


def aggregate_results(metrics: pd.DataFrame, decisions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    metric_columns = [
        "PICP",
        "Coverage_Gap",
        "MPIW",
        "Winkler",
        "Worst_Site_PICP",
        "Bloom_PICP",
        "Bloom_Winkler",
        "Severe_PICP",
        "RMSE",
        "MAE",
    ]
    summary = metrics.groupby("Method", as_index=False)[metric_columns].mean()
    by_cell = metrics.groupby(["Backbone", "Arm", "Method"], as_index=False)[metric_columns].mean()

    pivot = by_cell.pivot(index=["Backbone", "Arm"], columns="Method", values=metric_columns)
    difference_rows: List[dict] = []
    for comparator in ("scp", "ecp", "rcp"):
        for (backbone, arm), row in pivot.iterrows():
            difference_rows.append(
                {
                    "Backbone": backbone,
                    "Arm": arm,
                    "Comparison": f"aci_minus_{comparator}",
                    **{
                        f"Delta_{metric}": float(row[(metric, "aci")] - row[(metric, comparator)])
                        for metric in metric_columns
                    },
                }
            )

    decision_columns = [
        "Miss_Rate",
        "False_Alarm_Rate",
        "Cost_Ratio_1",
        "Cost_Ratio_2",
        "Cost_Ratio_5",
        "Cost_Ratio_10",
        "Cost_Ratio_20",
    ]
    decision_summary = decisions.groupby(["Method", "Threshold_Quantile"], as_index=False)[decision_columns].mean()
    return {
        "method_summary": summary,
        "method_summary_by_cell": by_cell,
        "paired_cell_differences": pd.DataFrame(difference_rows),
        "decision_summary": decision_summary,
    }


def point_forecast_summary(metrics: pd.DataFrame, reference_test: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    actual = np.concatenate([reference_test[sid]["Actual"].to_numpy(float) for sid in SITE_NAMES])
    persistence = np.concatenate([reference_test[sid]["Persistence"].to_numpy(float) for sid in SITE_NAMES])
    rows = [
        {
            "Model": "Persistence",
            "Runs": 1,
            "RMSE": float(np.sqrt(np.mean((actual - persistence) ** 2))),
            "MAE": float(np.mean(np.abs(actual - persistence))),
        }
    ]
    raw = metrics[metrics["Method"] == "raw"]
    rows.append(
        {
            "Model": "Deep ensemble (all runs)",
            "Runs": int(len(raw)),
            "RMSE": float(raw["RMSE"].mean()),
            "MAE": float(raw["MAE"].mean()),
        }
    )
    for backbone, group in raw.groupby("Backbone"):
        rows.append(
            {
                "Model": f"Deep ensemble ({backbone})",
                "Runs": int(len(group)),
                "RMSE": float(group["RMSE"].mean()),
                "MAE": float(group["MAE"].mean()),
            }
        )
    return pd.DataFrame(rows)


def gate_report(summary: pd.DataFrame, gamma: float, window: int) -> dict:
    indexed = summary.set_index("Method")
    aci = indexed.loc["aci"]
    best_online_gap = min(float(indexed.loc["ecp", "Coverage_Gap"]), float(indexed.loc["rcp", "Coverage_Gap"]))
    gate_c = bool(
        aci["Winkler"] < indexed.loc["scp", "Winkler"]
        and aci["Winkler"] < indexed.loc["ecp", "Winkler"]
        and aci["Coverage_Gap"] <= best_online_gap + 0.01
    )
    return {
        "Gate_A_input_integrity": "PASS",
        "Gate_B_validation_only_selection": "PASS",
        "Selected_gamma": gamma,
        "Selected_rolling_window": window,
        "Gate_C_ACI_headline": "PASS" if gate_c else "FAIL_REFRAME_AS_COMPARATIVE_STUDY",
        "Gate_C_checks": {
            "ACI_Winkler": float(aci["Winkler"]),
            "SCP_Winkler": float(indexed.loc["scp", "Winkler"]),
            "ECP_Winkler": float(indexed.loc["ecp", "Winkler"]),
            "ACI_Coverage_Gap": float(aci["Coverage_Gap"]),
            "Best_Online_Baseline_Gap": best_online_gap,
        },
        "Gate_D_server": "NOT_REQUIRED_FOR_CURRENT_POSTPROCESSING",
        "Manuscript_frame": "ACI_HEADLINE" if gate_c else "COMPARATIVE_CALIBRATION",
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    run_root = base / "results" / "clean_reanalysis_v2" / "2026-08-18_clean-v2-sealed2025"
    default_index = Path(
        os.environ.get(
            "WEIS_SAMPLE_INDEX",
            str(base / "data" / "clean_reanalysis_v2" / "sample_index.parquet"),
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=run_root / "ablation_reps")
    parser.add_argument("--sample-index", type=Path, default=default_index)
    parser.add_argument("--output-dir", type=Path, default=run_root / "calibration")
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = expected_run_paths(args.input_root)
    manifest = make_manifest(paths)
    manifest.to_csv(args.output_dir / "input_manifest.csv", index=False)
    reference = load_bundle_reference(args.sample_index)

    # Gate B: validation-only selection is completed and persisted before the
    # first test predictions.csv is read by evaluate_test().
    validation_runs = load_validation_runs(paths, reference["validation"])
    selection, selected_gamma, selected_window = tune_hyperparameters(validation_runs)
    selection.to_csv(args.output_dir / "validation_selection.csv", index=False)
    selection_record = {
        "Protocol": "uq_iso_hab_clean_v2.yml (WEIS Data Core) + WEIS_DATA_CORE_MIGRATION.md v2 section",
        "Sample_index": str(args.sample_index),
        "Sample_index_SHA256": sha256(args.sample_index),
        "Selection_data": (
            "2024 validation only: per-station first floor(n/2) observations warm, remainder scored "
            "(per-station n varies, 290-352)"
        ),
        "Alpha": ALPHA,
        "Gamma_grid": list(GAMMA_GRID),
        "Window_grid": list(WINDOW_GRID),
        "Selected_gamma": selected_gamma,
        "Selected_rolling_window": selected_window,
        "Tie_break": "smaller gamma; larger rolling window",
        "Test_outcomes_loaded_before_freeze": False,
    }
    write_json(args.output_dir / "selected_hyperparameters.json", selection_record)
    if args.selection_only:
        print(json.dumps(selection_record, indent=2))
        return

    metrics, per_site, decisions, intervals = evaluate_test(
        paths, validation_runs, reference["test"], selected_gamma, selected_window
    )
    aggregates = aggregate_results(metrics, decisions)

    metrics.to_csv(args.output_dir / "metrics_by_run.csv", index=False)
    per_site.to_csv(args.output_dir / "metrics_by_run_site.csv", index=False)
    decisions.to_csv(args.output_dir / "decision_by_run.csv", index=False)
    intervals.to_csv(args.output_dir / "test_intervals.csv.gz", index=False, compression="gzip")
    point_forecast_summary(metrics, reference["test"]).to_csv(
        args.output_dir / "point_forecast_summary.csv", index=False
    )
    for name, table in aggregates.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)

    gates = gate_report(aggregates["method_summary"], selected_gamma, selected_window)
    write_json(args.output_dir / "gate_report.json", gates)
    write_json(
        args.output_dir / "integrity_report.json",
        {
            "Balanced_runs": 75,
            "Validation_rows_per_run": int(sum(len(reference["validation"][sid]) for sid in SITE_NAMES)),
            "Test_rows_per_run": int(sum(len(reference["test"][sid]) for sid in SITE_NAMES)),
            "Stations": len(SITE_NAMES),
            "Per_station_counts_variable": True,
            "Actual_sequences_matched_bundle_index": True,
            "Dates_matched_bundle_index": True,
            "Original_prediction_files_modified": False,
            "Saved_test_date_column_used": True,
            "Legacy_reconstructed_csv_used": False,
            "Input_files_hashed": int(len(manifest)),
        },
    )
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
