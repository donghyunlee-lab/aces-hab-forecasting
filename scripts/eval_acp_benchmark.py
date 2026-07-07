"""Unified ACP benchmark — raw vs split-conformal vs online ACI, across reps.

Supersedes the demo (scripts/demo_online_acp.py) for the paper. For every
replicate r0..r5 and every (backbone, arm) that has BOTH test predictions
(predictions.csv) and 2023-val predictions (val_predictions.csv, produced by
dump_val_predictions.py), it evaluates three calibration methods on the 2024
test set at the 90% target:

  raw   : Gaussian mu +/- z*sigma (z=1.6449), no conformal correction.
  split : per-site split-conformal, quantile of nonconformity scores from the
          calibration source (default 2023 val => the FAIR baseline).
  aci   : per-site online ACI (Gibbs & Candes 2021, OnlineACP), warm-started
          with the same calibration scores; the quantile adapts through 2024.

Metrics per method: PICP, |gap|, worst-site PICP, site-PICP variance, MPIW,
MPIW@0.90 (coverage-matched), Winkler interval score, tail(bloom) PICP/Winkler.
CRPS is a property of (mu,sigma) only, so it is reported once per arm.

Reps are the statistical unit: results are aggregated to mean +/- 95% CI per
(backbone, arm, method).

Outputs (results/acp_benchmark/):
  master_metrics.csv   - one row per (rep, backbone, arm, method)
  master_summary.csv   - mean/sd/95%CI per (backbone, arm, method)
  per_site.csv         - per-site PICP per (backbone, arm, method), rep-mean
  gamma_sensitivity.csv- (only with --gamma-sweep) ACI picp/mpiw/winkler vs gamma

Usage:
  python scripts/eval_acp_benchmark.py --alpha 0.10 --gamma 0.02
  python scripts/eval_acp_benchmark.py --calib-source 2024head --calib-frac 0.2
  python scripts/eval_acp_benchmark.py --gamma-sweep 0.005 0.01 0.02 0.05 0.1
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.uq_scores import (winkler_score, crps_gaussian, matched_mpiw,
                                       conditional_coverage)
from src.evaluation.uncertainty import OnlineACP

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = f"{BASE}/results/acp_benchmark"
EPS = 1e-6
BACKBONES = ["Mamba", "GRU", "iTransformer"]
ARMS = ["Standard_NLL", "ISO_NLL", "BetaNLL_0.5", "BetaNLL_1.0", "Faithful"]
REPS = [0, 1, 2, 3, 4, 5]


def arm_dir(bb, arm, rep):
    if rep == 0:
        return f"{BASE}/results/ablation/{bb}_{arm}"
    return f"{BASE}/results/ablation_reps/r{rep}/{bb}_{arm}"


def z_for(alpha):
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def _site_metrics(y, mu, sd, lower, upper, alpha, bloom_thr, sids):
    """Bundle the per-method metrics on a (test) slice."""
    picp = float(np.mean((y >= lower) & (y <= upper)))
    cc = conditional_coverage(y, lower, upper, sids, alpha=alpha, bloom_thr=bloom_thr)
    return dict(
        PICP=picp, gap=abs(picp - (1 - alpha)),
        worst_site=cc["worst_site_picp"], site_var=cc["site_picp_var"],
        MPIW=float(np.mean(upper - lower)),
        MPIW_at_0p90=matched_mpiw(y, mu, lower, upper, target=1 - alpha),
        Winkler=winkler_score(y, lower, upper, alpha),
        tail_PICP=cc["tail_picp"], tail_Winkler=cc["tail_winkler"],
        tail_thr=cc["tail_thr"],
        per_site=cc["per_site_picp"],
    )


def methods_on(test, val, alpha, gamma, calib_source, calib_frac, dump_path=None):
    """Return {method: metrics} for one (bb, arm, rep). Evaluated per site.

    If dump_path is given, also write the per-step interval bounds (one row per
    test observation: Date x Station_ID) for raw/split/ACI to that CSV. These
    bounds are computed here anyway for the metrics; the dump just persists them
    so figures (interval-band time series) and the decision layer (per-step
    miss/false-alarm) can reuse them with NO retraining.
    """
    z = z_for(alpha)
    test = test.sort_values(["Station_ID", "Date"]).reset_index(drop=True)
    y = test["Actual"].to_numpy(float)
    mu = test["Predicted_Mean"].to_numpy(float)
    sd = test["Predicted_Std"].to_numpy(float)
    sids = test["Station_ID"].to_numpy()
    bloom_thr = float(np.quantile(y, 0.90))  # fixed tail threshold per arm/rep

    # accumulate per-method interval bounds aligned to `test` order
    raw_lo, raw_hi = mu - z * sd, mu + z * sd
    split_lo = np.empty_like(y); split_hi = np.empty_like(y)
    aci_lo = np.empty_like(y); aci_hi = np.empty_like(y)
    keep = np.ones(len(y), bool)  # all kept for val-calib; trimmed for 2024head

    for s in np.unique(sids):
        m = np.where(sids == s)[0]
        ys, mus, sds = y[m], mu[m], sd[m]

        k = 0  # number of head points consumed for calibration (2024head only)
        if calib_source == "val":
            v = val[val["Station_ID"] == s]
            if len(v) == 0:            # no calibration data for this site -> drop it
                keep[m] = False
                continue
            warm = (np.abs(v["Actual"].to_numpy(float) - v["Predicted_Mean"].to_numpy(float))
                    / (v["Predicted_Std"].to_numpy(float) + EPS))
            ev = np.arange(len(m))     # evaluate on ALL test points
        else:  # 2024head: calibrate on first calib_frac of this site's test
            k = max(5, int(round(calib_frac * len(m))))
            warm = np.abs(ys[:k] - mus[:k]) / (sds[:k] + EPS)
            keep[m[:k]] = False
            ev = np.arange(k, len(m))

        q = float(np.quantile(warm, 1 - alpha))
        split_lo[m] = mus - q * (sds + EPS)
        split_hi[m] = mus + q * (sds + EPS)

        res = OnlineACP(alpha=alpha, gamma=gamma).run_stream(
            ys[ev], mus[ev], sds[ev], warm_scores=warm)
        aci_lo[m[ev]] = res["lower"]; aci_hi[m[ev]] = res["upper"]
        if k:  # head points were consumed for calibration -> no ACI interval there
            aci_lo[m[:k]] = np.nan
            aci_hi[m[:k]] = np.nan

    kept = keep & np.isfinite(aci_lo)

    if dump_path is not None:
        pd.DataFrame({
            "Date": test["Date"].to_numpy(),
            "Station_ID": sids,
            "Actual": y, "Pred_Mean": mu, "Pred_Std": sd,
            "raw_lo": raw_lo, "raw_hi": raw_hi,
            "split_lo": split_lo, "split_hi": split_hi,
            "aci_lo": aci_lo, "aci_hi": aci_hi,
            "kept": kept,
        }).to_csv(dump_path, index=False)

    out = {}
    out["raw"] = _site_metrics(y[kept], mu[kept], sd[kept], raw_lo[kept], raw_hi[kept],
                               alpha, bloom_thr, sids[kept])
    out["split"] = _site_metrics(y[kept], mu[kept], sd[kept], split_lo[kept], split_hi[kept],
                                 alpha, bloom_thr, sids[kept])
    out["aci"] = _site_metrics(y[kept], mu[kept], sd[kept], aci_lo[kept], aci_hi[kept],
                               alpha, bloom_thr, sids[kept])
    crps = crps_gaussian(y[kept], mu[kept], sd[kept])
    for mth in out:
        out[mth]["CRPS"] = crps
    return out


def load_pair(bb, arm, rep):
    d = arm_dir(bb, arm, rep)
    tp, vp = f"{d}/predictions.csv", f"{d}/val_predictions.csv"
    if not (os.path.exists(tp) and os.path.exists(vp)):
        return None
    return pd.read_csv(tp), pd.read_csv(vp)


METRIC_KEYS = ["PICP", "gap", "worst_site", "site_var", "MPIW", "MPIW_at_0p90",
               "Winkler", "tail_PICP", "tail_Winkler", "CRPS"]


def ci95(vals):
    v = np.asarray(vals, float)
    n = len(v)
    m = float(v.mean()) if n else float("nan")
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    half = (stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return m, sd, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--calib-source", choices=["val", "2024head"], default="val")
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--reps", nargs="+", type=int, default=REPS)
    ap.add_argument("--backbones", nargs="*", default=BACKBONES)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--gamma-sweep", nargs="*", type=float, default=None)
    args = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    target = 1 - args.alpha
    # val (fair baseline) is canonical -> unsuffixed; other sources get a suffix
    sfx = "" if args.calib_source == "val" else f"_{args.calib_source}"

    # ----- gamma sensitivity mode (ACI only) -----
    if args.gamma_sweep:
        rows = []
        for g in args.gamma_sweep:
            picps, mpiws, winks, gaps = [], [], [], []
            for bb in args.backbones:
                for arm in args.arms:
                    for rep in args.reps:
                        pair = load_pair(bb, arm, rep)
                        if pair is None:
                            continue
                        r = methods_on(pair[0], pair[1], args.alpha, g,
                                       args.calib_source, args.calib_frac)["aci"]
                        picps.append(r["PICP"]); mpiws.append(r["MPIW"])
                        winks.append(r["Winkler"]); gaps.append(r["gap"])
            rows.append(dict(gamma=g, n=len(picps),
                             PICP_mean=np.mean(picps), gap_mean=np.mean(gaps),
                             MPIW_mean=np.mean(mpiws), Winkler_mean=np.mean(winks)))
            print(f"gamma={g:<6} PICP={np.mean(picps):.3f} "
                  f"|gap|={np.mean(gaps):.3f} MPIW={np.mean(mpiws):.2f} "
                  f"Winkler={np.mean(winks):.2f}  (n={len(picps)})")
        pd.DataFrame(rows).to_csv(f"{OUTD}/gamma_sensitivity.csv", index=False)
        print(f"\nSaved -> {OUTD}/gamma_sensitivity.csv")
        return

    # ----- main benchmark -----
    perstep_dir = f"{OUTD}/perstep{sfx}"
    os.makedirs(perstep_dir, exist_ok=True)
    rep_rows, site_rows = [], []
    for bb in args.backbones:
        for arm in args.arms:
            for rep in args.reps:
                pair = load_pair(bb, arm, rep)
                if pair is None:
                    continue
                res = methods_on(pair[0], pair[1], args.alpha, args.gamma,
                                 args.calib_source, args.calib_frac,
                                 dump_path=f"{perstep_dir}/{bb}_{arm}_r{rep}.csv")
                for mth, mvals in res.items():
                    row = dict(Backbone=bb, Arm=arm, Rep=rep, Method=mth)
                    row.update({k: mvals[k] for k in METRIC_KEYS})
                    rep_rows.append(row)
                    for sid, p in mvals["per_site"].items():
                        site_rows.append(dict(Backbone=bb, Arm=arm, Rep=rep,
                                              Method=mth, Station_ID=sid, PICP=p))
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(f"{OUTD}/master_metrics{sfx}.csv", index=False)

    # aggregate over reps
    summ = []
    for (bb, arm, mth), g in rep_df.groupby(["Backbone", "Arm", "Method"]):
        row = dict(Backbone=bb, Arm=arm, Method=mth, n_reps=len(g))
        for k in METRIC_KEYS:
            m, sd, half = ci95(g[k].to_numpy())
            row[f"{k}_mean"], row[f"{k}_sd"], row[f"{k}_ci95"] = m, sd, half
        summ.append(row)
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(f"{OUTD}/master_summary{sfx}.csv", index=False)

    site_df = pd.DataFrame(site_rows)
    (site_df.groupby(["Backbone", "Arm", "Method", "Station_ID"])["PICP"]
     .mean().reset_index().to_csv(f"{OUTD}/per_site{sfx}.csv", index=False))

    # console headline: target coverage recovery per method (pooled over arms/bb)
    print(f"\n=== ACP benchmark (target={target:.2f}, calib={args.calib_source}, "
          f"gamma={args.gamma}) — pooled mean over {rep_df['Arm'].nunique()} arms x "
          f"{rep_df['Backbone'].nunique()} backbones x reps ===")
    hdr = f"{'method':6} | {'PICP':>6} {'|gap|':>6} {'worst':>6} {'MPIW':>7} {'Winkler':>8} {'tailPICP':>8}"
    print(hdr); print("-" * len(hdr))
    for mth in ["raw", "split", "aci"]:
        g = rep_df[rep_df.Method == mth]
        print(f"{mth:6} | {g.PICP.mean():6.3f} {g.gap.mean():6.3f} "
              f"{g.worst_site.mean():6.3f} {g.MPIW.mean():7.2f} "
              f"{g.Winkler.mean():8.2f} {g.tail_PICP.mean():8.3f}")
    print(f"\nSaved -> {OUTD}/{{master_metrics,master_summary,per_site}}{sfx}.csv")


if __name__ == "__main__":
    main()
