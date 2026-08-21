"""Read-only adapter for the UQ-ISO-HAB clean WEIS Data Core bundles.

This module intentionally does not fall back to imputed_daily_data.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


STUDY_ID = "uq-iso-hab"
DEFAULT_ANALYSIS_ID = os.environ.get("UQ_WEIS_ANALYSIS_ID", "clean_reanalysis_v1")


def _default_core_root() -> Path:
    configured = os.environ.get("WEIS_DATA_CORE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "weis-data-core"


def bundle_dir(core_root: str | Path | None = None, analysis_id: str | None = None) -> Path:
    root = Path(core_root).resolve() if core_root else _default_core_root()
    return root / "data/bundles" / STUDY_ID / (analysis_id or DEFAULT_ANALYSIS_ID)


def load_clean_bundle(
    core_root: str | Path | None = None,
    split: str | None = None,
    *,
    analysis_id: str | None = None,
    verify: bool = True,
) -> Any:
    analysis = analysis_id or DEFAULT_ANALYSIS_ID
    root = Path(core_root).resolve() if core_root else _default_core_root()
    source_path = str(root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from weis_core.consumer import load_model_bundle

    bundle = load_model_bundle(root / "data/bundles" / STUDY_ID / analysis, verify=verify)
    manifest = bundle.manifest
    if manifest["study_id"] != STUDY_ID or manifest["analysis_id"] != analysis:
        raise ValueError("Unexpected WEIS bundle identity")
    if manifest["bundle_type"] != "forecast_windows":
        raise ValueError("UQ clean reanalysis requires a forecast-window bundle")
    if manifest["lookback_steps"] != 30 or manifest["horizons_steps"] != [1]:
        raise ValueError("UQ clean reanalysis expects lookback=30 and horizon=[1]")
    name, count = min(manifest["split_counts"].items(), key=lambda item: item[1])
    if count == 0:
        raise ValueError(f"Bundle split '{name}' is empty; refusing to serve a partial bundle")
    if split is None:
        return bundle
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return bundle.split(split)


def model_arrays(
    split: str,
    core_root: str | Path | None = None,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    """Return mask-aware inputs and observed-only targets for one split."""

    selected = load_clean_bundle(core_root, split, analysis_id=analysis_id)
    arrays = selected["arrays"]
    target_mask = np.asarray(arrays["y_observed_mask"], dtype=bool)
    if not target_mask.all():
        raise ValueError("UQ horizon-1 bundle unexpectedly contains an unobserved target")
    if len(selected["index"]) == 0:
        raise ValueError(f"Split '{split}' resolved to zero samples")
    return {
        "index": selected["index"],
        "x": arrays["x_scaled_zero"],
        "x_observed_mask": arrays["x_observed_mask"],
        "x_available_mask": arrays["x_available_mask"],
        "x_age_steps": arrays["x_age_steps"],
        "y": arrays["y_observed_raw"][:, 0],
        "y_scaled": arrays["y_scaled_zero"][:, 0],
        "y_mask": target_mask[:, 0],
    }


def target_scaler(
    core_root: str | Path | None = None,
    analysis_id: str | None = None,
) -> dict[str, float]:
    """Train-only global scaler for the horizon-1 target, for inverse transforms."""

    bundle = load_clean_bundle(core_root, analysis_id=analysis_id)
    entry = bundle.manifest["target_scaler"][0]
    if entry.get("horizon_steps") != 1:
        raise ValueError("Expected the horizon-1 target scaler first")
    return {"mean": float(entry["mean"]), "standard_deviation": float(entry["standard_deviation"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--analysis-id", default=None)
    args = parser.parse_args()
    analysis = args.analysis_id or DEFAULT_ANALYSIS_ID
    values = model_arrays(args.split, args.core_root, analysis)
    print(json.dumps({
        "study_id": STUDY_ID,
        "analysis_id": analysis,
        "split": args.split,
        "samples": len(values["index"]),
        "x_shape": list(values["x"].shape),
        "y_shape": list(values["y"].shape),
        "observed_input_fraction": float(values["x_observed_mask"].mean()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
