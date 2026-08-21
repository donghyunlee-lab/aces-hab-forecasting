#!/usr/bin/env python3
"""Fresh, resumable retraining driver for the Q1 redesign.

Every output is namespaced by ``--tag``.  Old checkpoints are never searched
or reused.  The driver fits the corrected train-only, core-11 preprocessor via
``run_experiment`` and exports complete 2023/2024 ensemble predictions for the
leakage-controlled calibration analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Required by PyTorch deterministic algorithms for CUDA >= 10.2. Set this
# before torch initializes a cuBLAS handle so seeded reruns are reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from scripts.run_ablation_loss import ARM_CONFIG, DEFAULT_EPOCHS  # noqa: E402
from scripts.run_experiment import create_config, run_experiment  # noqa: E402


BACKBONES = ("Mamba", "GRU", "iTransformer")
ARMS = ("Standard_NLL", "ISO_NLL", "BetaNLL_0.5", "BetaNLL_1.0", "Faithful")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def seeds_for(rep: int, members: int) -> list[int]:
    return [5 * rep + index for index in range(members)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=BASE.parent, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def environment_record(args) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "tag": args.tag,
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "reproducibility_note": (
            "RNGs and DataLoader order are seeded. PyTorch may warn when a CUDA "
            "kernel, including memory-efficient attention, lacks a deterministic implementation."
        ),
        "preprocessing": "train-only fit; nine physicochemical inputs plus two causal Chl-a derivatives",
        "train_period": ["2021-01-01", "2022-12-31"],
        "validation_period": ["2023-01-01", "2023-12-31"],
        "test_period": ["2024-01-01", "2024-12-31"],
    }


def write_source_manifest(result_root: Path) -> None:
    tracked = [
        BASE / "data" / "imputed_daily_data.csv",
        BASE / "src" / "data" / "preprocessor.py",
        BASE / "src" / "training" / "callbacks.py",
        BASE / "src" / "training" / "trainer.py",
        BASE / "scripts" / "run_experiment.py",
        BASE / "scripts" / "run_ablation_loss.py",
        BASE / "scripts" / "run_redesign_retrain.py",
        BASE / "scripts" / "eval_calibration_redesign.py",
    ]
    rows = [
        {"path": str(path.relative_to(BASE)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in tracked
    ]
    (result_root / "source_manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def member_path(model_root: Path, rep: int, backbone: str, arm: str, seed: int) -> Path:
    return model_root / f"r{rep}" / f"{backbone}_{arm}_seed{seed}.pth"


def completion_marker(path: Path) -> Path:
    return path.with_name(path.name + ".complete.json")


def marker_is_valid(path: Path) -> bool:
    marker = completion_marker(path)
    if not path.is_file() or not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("status") == "complete" and record.get("sha256") == sha256(path)


def mark_complete(path: Path, **metadata) -> None:
    record = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        **metadata,
    }
    completion_marker(path).write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def backfill_completion_markers(result_root: Path, model_root: Path) -> int:
    """Certify members belonging to already completed prediction pairs.

    Runs created before completion markers existed remain safely resumable:
    ``run_config.json`` records the exact checkpoint hashes used for each
    complete validation/test prediction pair. Any unmatched checkpoint is
    deliberately left uncertified and will be retrained.
    """
    certified = 0
    for config_path in sorted((result_root / "ablation_reps").glob("r*/*/run_config.json")):
        run_dir = config_path.parent
        if not (run_dir / "predictions.csv").is_file() or not (run_dir / "val_predictions.csv").is_file():
            continue
        record = json.loads(config_path.read_text(encoding="utf-8"))
        rep = int(record["rep"])
        backbone = str(record["backbone"])
        arm = str(record["arm"])
        expected_hashes = record.get("member_sha256", {})
        for raw_path, expected_hash in expected_hashes.items():
            path = Path(raw_path)
            if not path.is_file():
                path = model_root / f"r{rep}" / Path(raw_path).name
            if not path.is_file() or sha256(path) != expected_hash:
                raise RuntimeError(f"completed prediction pair has a missing or changed member: {path}")
            if not marker_is_valid(path):
                seed = int(path.stem.rsplit("seed", 1)[1])
                mark_complete(
                    path,
                    rep=rep,
                    backbone=backbone,
                    arm=arm,
                    seed=seed,
                    evidence=str(config_path.relative_to(result_root)),
                )
                certified += 1
    return certified


def train_members(args, model_root: Path, rep: int, backbone: str, arm: str) -> list[Path]:
    arm_config = ARM_CONFIG[arm]
    paths = []
    for seed in seeds_for(rep, args.members):
        path = member_path(model_root, rep, backbone, arm, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        paths.append(path)
        if marker_is_valid(path):
            print(f"[resume] r{rep} {backbone}/{arm} seed={seed}: {path}")
            continue
        if path.exists():
            print(f"[restart incomplete] r{rep} {backbone}/{arm} seed={seed}: {path}")

        config = create_config(model_type=backbone, uq_method=arm_config["train_uq"])
        config.update(
            {
                "core_features_only": True,
                "decoupled": True,
                "seed": seed,
                "epochs": 1 if args.test_run else args.epochs,
                "training_only": True,
                # HABTrainer currently constructs the TensorDataset from tensors
                # that already live on the selected accelerator. CUDA tensors
                # cannot be read safely from forked DataLoader workers, so keep
                # loading in the training process. The panel has only 3,500
                # training sequences, making worker parallelism unnecessary.
                "num_workers": 0,
                "model_save_path": str(path),
                "use_iso_nll": arm_config.get("use_iso_nll", False),
            }
        )
        for key in ("lambd_indep", "lambd_sharp", "beta_nll"):
            if key in arm_config:
                config[key] = arm_config[key]
        print(f"[train] r{rep} {backbone}/{arm} seed={seed} -> {path}")
        run_experiment(config)
        if not path.exists():
            raise RuntimeError(f"training completed without checkpoint: {path}")
        mark_complete(path, rep=rep, backbone=backbone, arm=arm, seed=seed, evidence="training_returned")
    return paths


def evaluate_ensemble(
    args, result_root: Path, rep: int, backbone: str, arm: str, members: list[Path]
) -> None:
    output = result_root / "ablation_reps" / f"r{rep}" / f"{backbone}_{arm}"
    output.mkdir(parents=True, exist_ok=True)
    test_path = output / "predictions.csv"
    val_path = output / "val_predictions.csv"
    if test_path.exists() and val_path.exists():
        print(f"[resume] complete predictions r{rep} {backbone}/{arm}")
        return

    config = create_config(model_type=backbone, uq_method="Ensemble")
    config.update(
        {
            "core_features_only": True,
            "decoupled": True,
            "ensemble_paths": [str(path) for path in members],
            "skip_training": True,
            "apply_acp": False,
            "shap_analysis": False,
            "results_save_path": str(test_path),
            "val_results_save_path": str(val_path),
            "global_metrics_path": str(output / "global_metrics.csv"),
            "site_metrics_path": str(output / "site_metrics.csv"),
            "plot_save_path": str(output / "plots" / "predictions.png"),
        }
    )
    print(f"[evaluate] r{rep} {backbone}/{arm} ({len(members)} members)")
    run_experiment(config)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "rep": rep,
                "backbone": backbone,
                "arm": arm,
                "member_paths": [str(path) for path in members],
                "member_sha256": {str(path): sha256(path) for path in members},
                "core_features_only": True,
                "preprocessor_fit_period": config["train_period"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_output_manifest(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "output_manifest.json"):
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (root / "output_manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def finalize_calibration(result_root: Path) -> None:
    script = BASE / "scripts" / "eval_calibration_redesign.py"
    input_root = result_root / "ablation_reps"
    output = result_root / "calibration_redesign"
    command = [
        sys.executable,
        str(script),
        "--input-root",
        str(input_root),
        "--output-dir",
        str(output),
    ]
    print("[finalize] " + " ".join(command))
    subprocess.run(command, cwd=BASE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="2026-07-16_train-only-core11")
    parser.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--members", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="run calibration analysis after all 75 runs")
    parser.add_argument("--allow-cpu", action="store_true", help="permit non-CUDA execution for local smoke tests")
    args = parser.parse_args()

    if not args.test_run and not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("Full retraining requires CUDA; use --allow-cpu only for an intentional smoke test")
    if args.finalize and not (
        set(args.reps) == {1, 2, 3, 4, 5}
        and set(args.backbones) == set(BACKBONES)
        and set(args.arms) == set(ARMS)
        and args.members == 5
    ):
        raise SystemExit("--finalize requires the complete 5-rep x 3-backbone x 5-arm x 5-member design")

    model_root = BASE / "models" / args.tag
    result_root = BASE / "results" / args.tag
    result_root.mkdir(parents=True, exist_ok=True)
    log_handle = (result_root / "run.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)
    environment = environment_record(args)
    environment_dir = result_root / "environment_runs"
    environment_dir.mkdir(parents=True, exist_ok=True)
    environment_path = environment_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_pid{os.getpid()}.json"
    environment_path.write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.finalize or not (result_root / "environment.json").exists():
        (result_root / "environment.json").write_text(
            json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    write_source_manifest(result_root)
    certified = backfill_completion_markers(result_root, model_root)
    if certified:
        print(f"[resume] certified {certified} legacy checkpoints from completed prediction pairs")

    jobs = [(rep, backbone, arm) for rep in args.reps for backbone in args.backbones for arm in args.arms]
    print(f"Planned ensemble jobs: {len(jobs)}; members/job={args.members}; tag={args.tag}")
    for rep, backbone, arm in jobs:
        members = train_members(args, model_root, rep, backbone, arm)
        evaluate_ensemble(args, result_root, rep, backbone, arm, members)

    if args.finalize:
        finalize_calibration(result_root)
    print(f"Completed jobs for tag={args.tag}; writing output manifest")
    sys.stdout.flush()
    log_handle.flush()
    write_output_manifest(result_root)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_handle.close()
    print(f"Completed tag={args.tag}; outputs={result_root}")


if __name__ == "__main__":
    main()
