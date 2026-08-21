#!/usr/bin/env python3
"""Clean-bundle retraining driver for the sealed-2025 two-basin redesign.

Trains the legacy backbone/loss grid on WEIS Data Core ``clean_reanalysis_v2``
arrays (mask-aware 27-channel inputs: 9 scaled values, 9 observation masks,
9 capped observation ages) and exports 2024 validation / 2025 test ensemble
prediction pairs for the calibration analysis.

The 2025 split is inference-export only: no selection, early stopping, or
hyperparameter choice reads it. Early stopping uses the 2024 validation split,
which the protocol designates as the selection year.

Never reads imputed_daily_data.csv; resume discipline (checkpoint completion
sidecars) is inherited from the legacy redesign driver.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from scripts.run_ablation_loss import ARM_CONFIG  # noqa: E402
from scripts.run_redesign_retrain import (  # noqa: E402
    Tee,
    git_commit,
    mark_complete,
    marker_is_valid,
    seeds_for,
    sha256,
)
from scripts.run_experiment import create_config  # noqa: E402
from src.data.weis_core_adapter import bundle_dir, model_arrays  # noqa: E402
from src.models.builder import get_model  # noqa: E402
from src.models.inference import predict_uncertainty  # noqa: E402
from src.training.trainer import HABTrainer  # noqa: E402
from src.utils.constants import get_device  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

ANALYSIS_ID = "clean_reanalysis_v2"
BACKBONES = ("Mamba", "GRU", "iTransformer")
ARMS = tuple(ARM_CONFIG)
AGE_CAP_STEPS = 30.0
SPLIT_PERIODS = {
    "train": ("2021-01-01", "2023-12-31"),
    "validation": ("2024-01-01", "2024-12-31"),
    "test": ("2025-01-01", "2025-12-31"),
}


def load_clean_dataset() -> dict:
    splits = {
        name: model_arrays(name, analysis_id=ANALYSIS_ID)
        for name in ("train", "validation", "test")
    }
    station_ids = sorted({sid for s in splits.values() for sid in s["index"]["station_id"]})
    mapping = {sid: position for position, sid in enumerate(station_ids)}
    names: dict[str, str] = {}
    for s in splits.values():
        for sid, name in zip(s["index"]["station_id"], s["index"]["station_name_raw"]):
            names.setdefault(sid, name)
    site_names = [names[sid] for sid in station_ids]

    # Target pipeline mirrors the manuscript generator (log1p -> min-max),
    # with the transform fitted on the train split only.
    train_log = np.log1p(np.asarray(splits["train"]["y"], dtype=np.float64))
    lo, hi = float(train_log.min()), float(train_log.max())
    if hi <= lo:
        raise ValueError("Degenerate train target range")

    def encode(y_raw: np.ndarray) -> np.ndarray:
        return ((np.log1p(np.asarray(y_raw, np.float64)) - lo) / (hi - lo)).astype(np.float32)

    dataset: dict = {
        "site_names": site_names,
        "target_scale": {"transform": "log1p_minmax_train_only", "log1p_min": lo, "log1p_max": hi},
    }
    for name, s in splits.items():
        # Age policy (pre-committed): cap at 30 steps and scale to [0, 1];
        # positions with no prior observation carry NaN in the bundle and are
        # mapped to the cap. observed_mask already separates them from fresh 0s.
        age = np.asarray(s["x_age_steps"], dtype=np.float32)
        age = np.where(np.isfinite(age), age, AGE_CAP_STEPS)
        age = np.clip(age, 0.0, AGE_CAP_STEPS) / AGE_CAP_STEPS
        x = np.concatenate(
            [
                np.asarray(s["x"], np.float32),
                np.asarray(s["x_observed_mask"], np.float32),
                age,
            ],
            axis=-1,
        )
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite model inputs in split '{name}'")
        index = s["index"]
        dataset[name] = {
            "X": x,
            "y_raw": np.asarray(s["y"], np.float64),
            "y": encode(s["y"])[:, None],
            "station_index": index["station_id"].map(mapping).to_numpy(np.int64),
            "dates": pd.to_datetime(index["target_time_h1"]).dt.strftime("%Y-%m-%d").to_numpy(),
        }
    return dataset


def decode_predictions(mean_scaled: np.ndarray, var_scaled: np.ndarray, scale: dict):
    span = scale["log1p_max"] - scale["log1p_min"]
    mean_log = mean_scaled * span + scale["log1p_min"]
    var_log = var_scaled * span**2
    mean_raw = np.expm1(mean_log)
    # Delta-method conversion, matching the legacy export.
    var_raw = var_log * (mean_raw + 1.0) ** 2
    return mean_raw, np.sqrt(np.maximum(var_raw, 1e-12))


def provenance_record(args, dataset) -> dict:
    bundle_root = bundle_dir(analysis_id=ANALYSIS_ID)
    core_root = bundle_root.parents[3]
    manifest_path = bundle_root / "bundle_manifest.json"
    protocol_path = core_root / "config" / "protocols" / "uq_iso_hab_clean_v2.yml"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "tag": args.tag,
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "study_id": "uq-iso-hab",
        "analysis_id": ANALYSIS_ID,
        "data_source": f"weis_core_bundle:{bundle_root}",
        "bundle_manifest_sha256": sha256(manifest_path),
        "protocol_sha256": sha256(protocol_path),
        "input_channels": "9 scaled values + 9 observation masks + 9 ages = 27; age = min(age,30)/30, never-observed -> 1.0",
        "target_scale": dataset["target_scale"],
        "site_names": dataset["site_names"],
        "split_periods": SPLIT_PERIODS,
        "selection_used_test": False,
        "selection_note": "2025 test split used for inference export only.",
    }


def member_config(args, backbone: str, arm: str, seed: int, save_path: Path, site_names) -> dict:
    arm_config = ARM_CONFIG[arm]
    config = create_config(model_type=backbone, uq_method=arm_config["train_uq"])
    config.update(
        {
            "site_names": site_names,
            "core_features_only": True,
            "decoupled": True,
            "seed": seed,
            "epochs": 1 if args.test_run else args.epochs,
            "num_workers": 0,
            "model_save_path": str(save_path),
            "use_iso_nll": arm_config.get("use_iso_nll", False),
            "train_period": SPLIT_PERIODS["train"],
            "val_period": SPLIT_PERIODS["validation"],
            "test_period": SPLIT_PERIODS["test"],
            "data_source": f"weis_core:{ANALYSIS_ID}",
        }
    )
    for key in ("lambd_indep", "lambd_sharp", "beta_nll"):
        if key in arm_config:
            config[key] = arm_config[key]
    return config


def member_path(model_root: Path, rep: int, backbone: str, arm: str, seed: int) -> Path:
    return model_root / f"r{rep}" / f"{backbone}_{arm}_seed{seed}.pth"


def train_members(args, tensors, dataset, model_root: Path, rep: int, backbone: str, arm: str):
    paths = []
    n_features = tensors["train"]["X"].shape[-1]
    n_stations = len(dataset["site_names"])
    for seed in seeds_for(rep, args.members):
        path = member_path(model_root, rep, backbone, arm, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        paths.append(path)
        if marker_is_valid(path):
            print(f"[resume] r{rep} {backbone}/{arm} seed={seed}: {path}")
            continue
        if path.exists():
            print(f"[restart incomplete] r{rep} {backbone}/{arm} seed={seed}: {path}")
        config = member_config(args, backbone, arm, seed, path, dataset["site_names"])
        set_seed(seed)
        model = get_model(config, n_features=n_features, n_stations=n_stations).to(tensors["device"])
        trainer = HABTrainer(model, config)
        print(f"[train] r{rep} {backbone}/{arm} seed={seed} -> {path}")
        trainer.train(
            tensors["train"]["X"],
            tensors["train"]["y"],
            tensors["validation"]["X"],
            tensors["validation"]["y"],
            tensors["train"]["sid"],
            tensors["validation"]["sid"],
        )
        if not path.exists():
            raise RuntimeError(f"training completed without checkpoint: {path}")
        mark_complete(path, rep=rep, backbone=backbone, arm=arm, seed=seed, evidence="training_returned")
    return paths


def export_split(frame_split, mean_raw, sigma_raw, site_names) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Date": frame_split["dates"],
            "Actual": frame_split["y_raw"],
            "Predicted_Mean": mean_raw,
            "Predicted_Std": sigma_raw,
            "CI_Lower": mean_raw - 1.96 * sigma_raw,
            "CI_Upper": mean_raw + 1.96 * sigma_raw,
            "Station_ID": frame_split["station_index"],
        }
    )
    frame["Site"] = frame["Station_ID"].map(dict(enumerate(site_names)))
    return frame


def evaluate_ensemble(args, tensors, dataset, result_root: Path, rep: int, backbone: str, arm: str, members):
    output = result_root / "ablation_reps" / f"r{rep}" / f"{backbone}_{arm}"
    output.mkdir(parents=True, exist_ok=True)
    test_path = output / "predictions.csv"
    val_path = output / "val_predictions.csv"
    if test_path.exists() and val_path.exists():
        print(f"[resume] complete predictions r{rep} {backbone}/{arm}")
        return

    n_features = tensors["train"]["X"].shape[-1]
    n_stations = len(dataset["site_names"])
    arm_config = ARM_CONFIG[arm]
    config = create_config(model_type=backbone, uq_method=arm_config["train_uq"])
    config.update({"site_names": dataset["site_names"], "use_iso_nll": arm_config.get("use_iso_nll", False)})
    models = []
    for path in members:
        model = get_model(config, n_features=n_features, n_stations=n_stations).to(tensors["device"])
        model.load_state_dict(torch.load(path, map_location=tensors["device"]))
        model.eval()
        models.append(model)

    print(f"[evaluate] r{rep} {backbone}/{arm} ({len(models)} members)")
    exports = {}
    for split, out_path in (("validation", val_path), ("test", test_path)):
        mean, var = predict_uncertainty(
            models, tensors[split]["X"], station_ids=tensors[split]["sid"], method="Ensemble"
        )
        mean_raw, sigma_raw = decode_predictions(
            mean.cpu().numpy().squeeze(1), var.cpu().numpy().squeeze(1), dataset["target_scale"]
        )
        frame = export_split(dataset[split], mean_raw, sigma_raw, dataset["site_names"])
        frame.to_csv(out_path, index=False)
        exports[split] = {"path": str(out_path), "rows": int(len(frame))}

    (output / "run_config.json").write_text(
        json.dumps(
            {
                "rep": rep,
                "backbone": backbone,
                "arm": arm,
                "analysis_id": ANALYSIS_ID,
                "member_paths": [str(path) for path in members],
                "member_sha256": {str(path): sha256(path) for path in members},
                "exports": exports,
                "target_scale": dataset["target_scale"],
                "split_periods": SPLIT_PERIODS,
                "selection_used_test": False,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()

    result_root = BASE / "results" / ANALYSIS_ID / args.tag
    model_root = result_root / "models"
    result_root.mkdir(parents=True, exist_ok=True)
    log_path = result_root / f"driver_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    log_handle = log_path.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)

    device = get_device()
    print(f"[clean-v2] device={device} tag={args.tag}")
    dataset = load_clean_dataset()
    print(
        "[clean-v2] samples:",
        {name: len(dataset[name]["y_raw"]) for name in ("train", "validation", "test")},
        "stations:",
        dataset["site_names"],
    )
    tensors = {"device": device}
    for name in ("train", "validation", "test"):
        tensors[name] = {
            "X": torch.from_numpy(dataset[name]["X"]).float().to(device),
            "y": torch.from_numpy(dataset[name]["y"]).float().to(device),
            "sid": torch.from_numpy(dataset[name]["station_index"]).long().to(device),
        }

    manifest_path = result_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(provenance_record(args, dataset), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for rep in args.reps:
        for backbone in args.backbones:
            for arm in args.arms:
                members = train_members(args, tensors, dataset, model_root, rep, backbone, arm)
                evaluate_ensemble(args, tensors, dataset, result_root, rep, backbone, arm, members)

    print("[clean-v2] driver complete")


if __name__ == "__main__":
    main()
