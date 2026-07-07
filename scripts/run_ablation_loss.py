"""
Reproducible loss-ablation driver (E2 head-to-head + E3 cross-backbone).

Strengthens the novelty defense of ISO-NLL's P_corr against published remedies
(beta-NLL [Seitzer 2022], faithful heteroscedastic regression [Stirn 2023]) by
running every arm under an IDENTICAL pipeline: decoupled architecture,
M=5 deep ensemble over fixed seeds, and adaptive conformal prediction (ACP) at
the 0.90 target. See paper/revision/20260601-ablation-loss-comparison.md.

Design notes
------------
- "Standard NLL" is obtained via the ISO-NLL path with lambda_indep = 0 (pure
  Gaussian NLL), NOT the legacy adaptive-decorrelation branch.
- "ISO-NLL" uses lambda_indep = 0.05, lambda_sharp = 0.0 — matching the
  manuscript's stated method (sharpness comes from ACP, not a width loss term).
- beta-NLL / Faithful are selected by the member's uq_method string, which the
  trainer dispatches to the new loss functions (src/models/losses.py).
- Reproducibility: each member is trained under a fixed seed via
  run_experiment -> set_seed(config['seed']). Seeds: SEEDS below.

Usage (server)
--------------
    python scripts/run_ablation_loss.py                 # full E2 + E3
    python scripts/run_ablation_loss.py --block E2       # only Mamba x all arms
    python scripts/run_ablation_loss.py --test-run       # 1-epoch smoke test
    python scripts/run_ablation_loss.py --arms ISO_NLL Faithful --backbones Mamba

Outputs
-------
    results/ablation/<Backbone>_<Arm>/metrics.csv   (per arm, ACP @ 0.90)
    results/ablation/summary.csv                    (all arms, key metrics)
"""

import os
import sys
import argparse

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_experiment import run_experiment, create_config

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fixed seeds = ensemble members. Reproducible given the seeding infrastructure.
SEEDS = [0, 1, 2, 3, 4]

# Primary coverage target (alpha = 0.10 -> 0.90), matching the manuscript.
ACP_ALPHA = 0.10

DEFAULT_EPOCHS = 70

# Per-arm training configuration. `train_uq` is the member's uq_method, which
# the trainer uses to dispatch the loss.
ARM_CONFIG = {
    "Standard_NLL": {"train_uq": "ISONLL", "use_iso_nll": True,  "lambd_indep": 0.0,  "lambd_sharp": 0.0},
    "ISO_NLL":      {"train_uq": "ISONLL", "use_iso_nll": True,  "lambd_indep": 0.05, "lambd_sharp": 0.0},
    "BetaNLL_0.5":  {"train_uq": "BetaNLL", "use_iso_nll": False, "beta_nll": 0.5},
    "BetaNLL_1.0":  {"train_uq": "BetaNLL", "use_iso_nll": False, "beta_nll": 1.0},
    "Faithful":     {"train_uq": "Faithful", "use_iso_nll": False},
}

# E2: loss head-to-head on the Mamba backbone.
E2 = [("Mamba", arm) for arm in
      ["Standard_NLL", "BetaNLL_0.5", "BetaNLL_1.0", "Faithful", "ISO_NLL"]]
# E3: cross-backbone generality of P_corr (Mamba covered by E2).
E3 = [(bb, arm) for bb in ["GRU", "iTransformer"]
      for arm in ["Standard_NLL", "ISO_NLL"]]


def _member_path(backbone, arm, seed):
    d = f"{BASE}/models/ablation"
    os.makedirs(d, exist_ok=True)
    return f"{d}/{backbone}_{arm}_seed{seed}.pth"


def train_members(backbone, arm, epochs, test_run):
    """Train (or reuse) the M=5 ensemble members for one (backbone, arm)."""
    cfg_arm = ARM_CONFIG[arm]
    paths = []
    for seed in SEEDS:
        mc = create_config(model_type=backbone, uq_method=cfg_arm["train_uq"])
        mc["decoupled"] = True
        mc["seed"] = seed
        mc["epochs"] = 1 if test_run else epochs
        mc["use_iso_nll"] = cfg_arm.get("use_iso_nll", False)
        for k in ("lambd_indep", "lambd_sharp", "beta_nll"):
            if k in cfg_arm:
                mc[k] = cfg_arm[k]
        path = _member_path(backbone, arm, seed)
        mc["model_save_path"] = path
        if os.path.exists(path):
            mc["skip_training"] = True
            print(f"  [reuse] {backbone}/{arm} seed={seed}")
        else:
            print(f"  [train] {backbone}/{arm} seed={seed}")
        run_experiment(mc)
        paths.append(path)
    return paths


def eval_ensemble(backbone, arm, paths):
    """Evaluate the deep ensemble + ACP for one (backbone, arm)."""
    out = f"{BASE}/results/ablation/{backbone}_{arm}"
    os.makedirs(out, exist_ok=True)

    cfg = create_config(model_type=backbone, uq_method="Ensemble")
    cfg["decoupled"] = True
    cfg["ensemble_paths"] = paths
    cfg["skip_training"] = True
    cfg["apply_acp"] = True
    cfg["acp_alpha"] = ACP_ALPHA
    cfg["global_metrics_path"] = f"{out}/global_metrics.csv"
    cfg["site_metrics_path"] = f"{out}/site_metrics.csv"

    _, global_metrics, _, _ = run_experiment(cfg)

    gm = global_metrics.copy()
    gm["Backbone"] = backbone
    gm["Arm"] = arm
    gm["ACP_alpha"] = ACP_ALPHA
    gm.to_csv(f"{out}/metrics.csv", index=False)
    print(f"  [saved] {out}/metrics.csv")
    return gm


def run_job(backbone, arm, epochs, test_run):
    print("\n" + "=" * 60)
    print(f"ABLATION  backbone={backbone}  arm={arm}  seeds={SEEDS}")
    print("=" * 60)
    paths = train_members(backbone, arm, epochs, test_run)
    return eval_ensemble(backbone, arm, paths)


def _metric(gm, name):
    rows = gm[gm["Metric"] == name]["Value"].values
    return float(rows[0]) if len(rows) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["E2", "E3", "all"], default="all")
    ap.add_argument("--backbones", nargs="*", default=None,
                    help="Restrict to these backbones (overrides --block selection)")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="Restrict to these arms (overrides --block selection)")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--test-run", action="store_true")
    args = ap.parse_args()

    if args.block == "E2":
        jobs = list(E2)
    elif args.block == "E3":
        jobs = list(E3)
    else:
        jobs = list(dict.fromkeys(E2 + E3))  # dedup, preserve order

    if args.backbones:
        jobs = [(b, a) for (b, a) in jobs if b in args.backbones]
    if args.arms:
        jobs = [(b, a) for (b, a) in jobs if a in args.arms]

    if not jobs:
        print("No jobs match the given --block/--backbones/--arms filters.")
        return

    print(f"Planned jobs ({len(jobs)}): {jobs}")

    summary = []
    for backbone, arm in jobs:
        gm = run_job(backbone, arm, args.epochs, args.test_run)
        summary.append({
            "Backbone": backbone, "Arm": arm,
            "RMSE": _metric(gm, "RMSE"), "R2": _metric(gm, "R2"),
            "PICP": _metric(gm, "PICP"), "MPIW": _metric(gm, "MPIW"),
            "CWC": _metric(gm, "CWC"), "ACP_alpha": ACP_ALPHA,
        })

    out_dir = f"{BASE}/results/ablation"
    os.makedirs(out_dir, exist_ok=True)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{out_dir}/summary.csv", index=False)
    print("\n[Ablation Summary]")
    print(summary_df.to_string(index=False))
    print(f"\nSaved -> {out_dir}/summary.csv")
    print("NOTE: per-seed numbers will differ from the pre-seeding manuscript "
          "values; refresh paper numbers from these reproducible runs.")


if __name__ == "__main__":
    main()
