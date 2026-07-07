"""Dump 2023-VALIDATION predictions for every saved ensemble (inference only).

Why: the ACP benchmark needs a FAIR split-conformal baseline calibrated on a
held-out year (2023 val), not on the first part of the 2024 test year. The
trained pipeline only ever writes 2024-test predictions, so this standalone
driver reloads the saved .pth ensembles and runs them on the val split,
reusing the EXACT denormalisation recipe from run_experiment STEP 5.

It does NOT call run_experiment and does NOT train — only `predict_uncertainty`
(Ensemble path) + the preprocessor. The science (loss/seed/ACP) is untouched.

Output: <arm_dir>/val_predictions.csv with columns
        Actual, Predicted_Mean, Predicted_Std, Station_ID
where arm_dir is results/ablation/{bb}_{arm} (rep 0) or
results/ablation_reps/r{r}/{bb}_{arm} (rep>=1).

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True \
      python scripts/dump_val_predictions.py --reps 0 1 2 3 4 5
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# importing run_experiment also runs `torch.backends.cudnn.enabled = False`
# (needed for GRU on this env) — same backend state as the trained pipeline.
from scripts.run_experiment import create_config
from src.data.preprocessor import DataPreprocessor
from src.models.builder import get_model
from src.models.inference import predict_uncertainty
from src.utils.constants import get_device
from src.utils.reproducibility import set_seed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKBONES = ["Mamba", "GRU", "iTransformer"]
ARMS = ["Standard_NLL", "ISO_NLL", "BetaNLL_0.5", "BetaNLL_1.0", "Faithful"]


def seeds_for(rep):
    return list(range(5)) if rep == 0 else list(range(5 * rep, 5 * rep + 5))


def member_paths(bb, arm, rep):
    if rep == 0:
        return [f"{BASE}/models/ablation/{bb}_{arm}_seed{s}.pth" for s in seeds_for(rep)]
    return [f"{BASE}/models/ablation_reps/{bb}_{arm}_r{rep}_seed{s}.pth" for s in seeds_for(rep)]


def arm_dir(bb, arm, rep):
    if rep == 0:
        return f"{BASE}/results/ablation/{bb}_{arm}"
    return f"{BASE}/results/ablation_reps/r{rep}/{bb}_{arm}"


def prepare_val_data(device):
    """Build the preprocessor + val split once (shared across all ensembles)."""
    cfg = create_config(model_type="Mamba", uq_method="Ensemble")
    pre = DataPreprocessor(
        data_path=cfg["data_path"], site_names=cfg["site_names"],
        apply_log_transform=cfg.get("apply_log_transform", True),
        use_panel_data=cfg.get("use_panel_data", True),
    )
    df = pre.load_data()
    dfn, feats = pre.preprocess(df)
    ds = pre.split_train_test(
        df, dfn, seq_len=cfg["seq_len"], train_period=cfg["train_period"],
        val_period=cfg["val_period"], test_period=cfg["test_period"],
    )
    X_val = torch.from_numpy(ds["X_val"]).float().to(device)
    sid = torch.from_numpy(ds["val_station_ids"]).long().to(device)
    y_val = ds["y_val"].squeeze(-1) if ds["y_val"].ndim > 1 else ds["y_val"]
    return pre, feats, X_val, sid, np.asarray(y_val, float), ds["val_station_ids"]


def denorm_mean_std(pre, n_feat, mean_np, var_np):
    """Replicate run_experiment.py STEP 5 mean/variance denormalisation."""
    ti = pre.target_idx
    rng = pre.scaler.data_min_[ti], pre.scaler.data_max_[ti]
    data_range = rng[1] - rng[0]
    var_d = var_np * (data_range ** 2)
    dummy = np.zeros((len(mean_np), n_feat)); dummy[:, ti] = mean_np
    mean_d = pre.inverse_transform(dummy)[:, ti]
    if pre.apply_log_transform:
        var_d = var_d * ((mean_d + 1) ** 2)  # delta-method approx (same as pipeline)
    return mean_d, np.sqrt(var_d)


def denorm_target(pre, n_feat, y_np):
    ti = pre.target_idx
    dummy = np.zeros((len(y_np), n_feat)); dummy[:, ti] = y_np
    return pre.inverse_transform(dummy)[:, ti]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--backbones", nargs="*", default=BACKBONES)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--overwrite", action="store_true",
                    help="re-dump even if val_predictions.csv already exists")
    args = ap.parse_args()

    set_seed(0)  # ensemble eval is deterministic; fix RNG for good measure
    device = get_device()
    print(f"Using device: {device}")
    pre, feats, X_val, sid, y_val_norm, sid_np = prepare_val_data(device)
    n_feat = len(feats)
    y_val_denorm = denorm_target(pre, n_feat, y_val_norm)
    print(f"Val set: {len(y_val_norm)} samples, {len(np.unique(sid_np))} sites, "
          f"{n_feat} features. Target denorm range "
          f"[{y_val_denorm.min():.2f}, {y_val_denorm.max():.2f}] mg/m^3")

    done, skipped, missing = 0, 0, []
    # Reuse 5 model shells per backbone; just load_state_dict per ensemble.
    for bb in args.backbones:
        n_stations = 5
        cfg = create_config(model_type=bb, uq_method="Ensemble")
        cfg["decoupled"] = True
        shells = [get_model(cfg, n_features=n_feat, n_stations=n_stations).to(device)
                  for _ in range(5)]
        for rep in args.reps:
            for arm in args.arms:
                d = arm_dir(bb, arm, rep)
                if not os.path.isdir(d):
                    continue  # this (bb,arm,rep) ensemble was never produced
                out_csv = f"{d}/val_predictions.csv"
                if os.path.exists(out_csv) and not args.overwrite:
                    skipped += 1
                    continue
                paths = member_paths(bb, arm, rep)
                if not all(os.path.exists(p) for p in paths):
                    missing.append(f"r{rep} {bb}/{arm}")
                    continue
                models = []
                for shell, p in zip(shells, paths):
                    shell.load_state_dict(torch.load(p, map_location=device))
                    shell.eval()
                    models.append(shell)
                pm, pv = predict_uncertainty(models, X_val, station_ids=sid,
                                             method="Ensemble")
                mean_np = pm.cpu().numpy().squeeze(1)
                var_np = pv.cpu().numpy().squeeze(1)
                mean_d, std_d = denorm_mean_std(pre, n_feat, mean_np, var_np)
                pd.DataFrame({
                    "Actual": y_val_denorm,
                    "Predicted_Mean": mean_d,
                    "Predicted_Std": std_d,
                    "Station_ID": sid_np,
                }).to_csv(out_csv, index=False)
                done += 1
                print(f"  [val] r{rep} {bb}/{arm}  -> {out_csv}  "
                      f"(mu[{mean_d.min():.1f},{mean_d.max():.1f}] "
                      f"sd[{std_d.min():.2f},{std_d.max():.2f}])")

    print(f"\nDone. dumped={done} skipped(existing)={skipped} missing={len(missing)}")
    if missing:
        print("  missing members for:", ", ".join(missing))


if __name__ == "__main__":
    main()
