"""
Ensemble-level replication driver for the ISO_NLL-vs-Standard_NLL significance
study. Trains R independent M=5 deep ensembles per (backbone, arm), each ensemble
using a DISTINCT seed set, so between-ensemble (seed/init) variance can be
estimated and ISO vs Standard tested with a paired test.

Science is UNTOUCHED: this reuses ARM_CONFIG / create_config / run_experiment from
the validated pipeline; only the seed sets and output paths differ.

Seed set for replicate r: seeds = [5*r, 5*r+1, ..., 5*r+4].
  r=0 = {0..4}  -> the ALREADY-RUN ensemble (results/ablation/, models/ablation/);
                  reused, never overwritten (this driver refuses r=0).
  r>=1          -> models/ablation_reps/{bb}_{arm}_r{r}_seed{s}.pth
                   results/ablation_reps/r{r}/{bb}_{arm}/

Usage
-----
    python scripts/run_replicates.py --reps 1 2 3 4               # full study
    python scripts/run_replicates.py --reps 1 --backbones Mamba --arms ISO_NLL --test-run
"""
import os, sys, argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_experiment import run_experiment, create_config
from scripts.run_ablation_loss import ARM_CONFIG, ACP_ALPHA, DEFAULT_EPOCHS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BACKBONES = ["Mamba", "GRU", "iTransformer"]
ARMS = ["Standard_NLL", "ISO_NLL"]


def seeds_for(rep):
    return [5 * rep + i for i in range(5)]


def member_path(backbone, arm, rep, seed):
    d = f"{BASE}/models/ablation_reps"
    os.makedirs(d, exist_ok=True)
    return f"{d}/{backbone}_{arm}_r{rep}_seed{seed}.pth"


def out_dir(backbone, arm, rep):
    d = f"{BASE}/results/ablation_reps/r{rep}/{backbone}_{arm}"
    os.makedirs(d, exist_ok=True)
    return d


def train_members(backbone, arm, rep, epochs, test_run):
    cfg_arm = ARM_CONFIG[arm]
    paths = []
    for seed in seeds_for(rep):
        mc = create_config(model_type=backbone, uq_method=cfg_arm["train_uq"])
        mc["decoupled"] = True
        mc["seed"] = seed
        mc["epochs"] = 1 if test_run else epochs
        mc["use_iso_nll"] = cfg_arm.get("use_iso_nll", False)
        for k in ("lambd_indep", "lambd_sharp", "beta_nll"):
            if k in cfg_arm:
                mc[k] = cfg_arm[k]
        path = member_path(backbone, arm, rep, seed)
        mc["model_save_path"] = path
        if os.path.exists(path):
            mc["skip_training"] = True
            print(f"  [reuse] r{rep} {backbone}/{arm} seed={seed}")
        else:
            print(f"  [train] r{rep} {backbone}/{arm} seed={seed}")
        run_experiment(mc)
        paths.append(path)
    return paths


def eval_ensemble(backbone, arm, rep, paths):
    out = out_dir(backbone, arm, rep)
    cfg = create_config(model_type=backbone, uq_method="Ensemble")
    cfg["decoupled"] = True
    cfg["ensemble_paths"] = paths
    cfg["skip_training"] = True
    cfg["apply_acp"] = True
    cfg["acp_alpha"] = ACP_ALPHA
    cfg["global_metrics_path"] = f"{out}/global_metrics.csv"
    cfg["site_metrics_path"] = f"{out}/site_metrics.csv"
    cfg["results_save_path"] = f"{out}/predictions.csv"
    _, global_metrics, _, _ = run_experiment(cfg)
    gm = global_metrics.copy()
    gm["Backbone"] = backbone
    gm["Arm"] = arm
    gm["Rep"] = rep
    gm["ACP_alpha"] = ACP_ALPHA
    gm.to_csv(f"{out}/metrics.csv", index=False)
    print(f"  [saved] {out}/metrics.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--backbones", nargs="*", default=BACKBONES)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--test-run", action="store_true")
    args = ap.parse_args()

    if 0 in args.reps:
        sys.exit("Refusing rep 0 (= the existing results/ablation run). Use reps >= 1.")

    jobs = [(r, b, a) for r in args.reps for b in args.backbones for a in args.arms]
    print(f"Planned replicate jobs ({len(jobs)}): reps={args.reps} "
          f"backbones={args.backbones} arms={args.arms} "
          f"epochs={'1 (TEST)' if args.test_run else args.epochs}")
    for rep, b, a in jobs:
        print("\n" + "=" * 60)
        print(f"REPLICATE r={rep}  backbone={b}  arm={a}  seeds={seeds_for(rep)}")
        print("=" * 60)
        paths = train_members(b, a, rep, args.epochs, args.test_run)
        eval_ensemble(b, a, rep, paths)
    print("\nDone.")


if __name__ == "__main__":
    main()
