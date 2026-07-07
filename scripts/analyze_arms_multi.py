"""Multi-arm loss statistics (Phase 4) + method comparison.

Extends the 2-arm pairwise design of scripts/analyze_replicates.py to the full
5-arm loss family, per backbone, and adds the confirmatory equivalence test
that the pre-registration (EXPERIMENT_DESIGN.md sec 4) requires for H1/H3.

Three analyses, all with the replicate (rep) as the statistical unit:

 1. LOSS family (per backbone), primary endpoints:
      RMSE         (accuracy, recomputed from test predictions)
      MPIW@0.90    (sharpness of the raw UQ, coverage-matched; from master_metrics, method=raw)
      tail_PICP    (bloom coverage under the deployed ACI; from master_metrics, method=aci)
    - Friedman omnibus across the 5 arms.
    - If significant: Wilcoxon each arm vs Standard_NLL, Holm-corrected.
    - TOST equivalence vs Standard (margin: RMSE +/-5% of Standard mean, PICP +/-0.02)
      -> "ISO ≈ Standard" (H3) and "all arms ≈ after calibration" (H1) are shown by
      EQUIVALENCE, not by a non-significant null.

 2. METHOD comparison: split vs online ACI, paired over (bb, arm, rep), on
      |gap|, tail_PICP, Winkler, MPIW  -> significance of the ACI gain.

Outputs (results/acp_benchmark/): loss_stats.csv, equivalence_tost.csv, method_compare.csv

Usage: python scripts/analyze_arms_multi.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = f"{BASE}/results/acp_benchmark"
BACKBONES = ["Mamba", "GRU", "iTransformer"]
ARMS = ["Standard_NLL", "ISO_NLL", "BetaNLL_0.5", "BetaNLL_1.0", "Faithful"]
REF = "Standard_NLL"


def arm_dir(bb, arm, rep):
    if rep == 0:
        return f"{BASE}/results/ablation/{bb}_{arm}"
    return f"{BASE}/results/ablation_reps/r{rep}/{bb}_{arm}"


def rmse_of(bb, arm, rep):
    p = f"{arm_dir(bb, arm, rep)}/predictions.csv"
    if not os.path.exists(p):
        return np.nan
    d = pd.read_csv(p)
    return float(np.sqrt(np.mean((d["Actual"] - d["Predicted_Mean"]) ** 2)))


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    k = len(p)
    adj = np.empty(k)
    running = 0.0
    for i, idx in enumerate(order):
        val = (k - i) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def tost(diff, margin):
    """Paired TOST. diff over reps, equivalence margin +/-margin.
    Returns (p_tost, equivalent_bool, ci90_lo, ci90_hi)."""
    d = np.asarray(diff, float)
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    if not (n > 1 and sd > 0):
        return float("nan"), False, float("nan"), float("nan")
    se = sd / np.sqrt(n)
    t_lo = (m + margin) / se      # H0: mean <= -margin
    t_hi = (m - margin) / se      # H0: mean >= +margin
    p_lo = stats.t.sf(t_lo, n - 1)   # upper tail
    p_hi = stats.t.cdf(t_hi, n - 1)  # lower tail
    p_tost = max(p_lo, p_hi)
    tc = stats.t.ppf(0.95, n - 1)    # 90% CI <=> TOST at 0.05
    ci_lo, ci_hi = m - tc * se, m + tc * se
    equiv = (ci_lo > -margin) and (ci_hi < margin)
    return float(p_tost), bool(equiv), float(ci_lo), float(ci_hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    reps = args.reps

    master = pd.read_csv(f"{OUTD}/master_metrics.csv")

    def metric_series(bb, arm, key, method):
        """Paired (over reps) array of `key` for method, from master_metrics."""
        sub = master[(master.Backbone == bb) & (master.Arm == arm)
                     & (master.Method == method) & (master.Rep.isin(reps))]
        sub = sub.set_index("Rep").reindex(reps)
        return sub[key].to_numpy(float)

    ENDPOINTS = [("RMSE", None), ("MPIW_at_0p90", "raw"), ("tail_PICP", "aci")]
    loss_rows, tost_rows = [], []

    for bb in BACKBONES:
        # build per-arm paired arrays
        data = {}
        for arm in ARMS:
            data[arm] = {}
            data[arm]["RMSE"] = np.array([rmse_of(bb, arm, r) for r in reps], float)
            data[arm]["MPIW_at_0p90"] = metric_series(bb, arm, "MPIW_at_0p90", "raw")
            data[arm]["tail_PICP"] = metric_series(bb, arm, "tail_PICP", "aci")

        for key, _m in ENDPOINTS:
            samples = [data[a][key] for a in ARMS]
            ok = np.all([np.isfinite(s).all() for s in samples])
            if ok and len(reps) >= 3:
                chi, pf = stats.friedmanchisquare(*samples)
            else:
                chi, pf = float("nan"), float("nan")
            # post-hoc vs Standard
            raw_p, contrasts = [], []
            for arm in ARMS:
                if arm == REF:
                    continue
                diff = data[arm][key] - data[REF][key]
                try:
                    _, pw = stats.wilcoxon(diff)
                except Exception:
                    pw = float("nan")
                raw_p.append(pw); contrasts.append(arm)
            adj_p = holm(raw_p)
            loss_rows.append(dict(
                Backbone=bb, metric=key, n=len(reps),
                friedman_chi=float(chi), friedman_p=float(pf),
                **{f"p_{a}": rp for a, rp in zip(contrasts, raw_p)},
                **{f"holm_{a}": ap_ for a, ap_ in zip(contrasts, adj_p)}))

            # TOST equivalence vs Standard
            ref_mean = float(np.nanmean(data[REF][key]))
            margin = 0.05 * abs(ref_mean) if key != "tail_PICP" else 0.02
            for arm in ARMS:
                if arm == REF:
                    continue
                diff = data[arm][key] - data[REF][key]
                pt, eq, lo, hi = tost(diff, margin)
                tost_rows.append(dict(Backbone=bb, metric=key, arm=arm, ref=REF,
                                      mean_diff=float(np.nanmean(diff)), margin=margin,
                                      ci90_lo=lo, ci90_hi=hi, p_tost=pt,
                                      equivalent=eq))

    pd.DataFrame(loss_rows).to_csv(f"{OUTD}/loss_stats.csv", index=False)
    pd.DataFrame(tost_rows).to_csv(f"{OUTD}/equivalence_tost.csv", index=False)

    # ---- method comparison: split vs ACI ----
    # UNIT = (backbone, arm) model configuration. Pooling every (bb, arm, rep) diff
    # would PSEUDOREPLICATE: arms share the same 2024 test year and reps share the
    # architecture, so those diffs are not independent and inflate the t-test df.
    # Instead average the paired diff over reps WITHIN each cell -> 15 independent
    # configurations, then test across them (Wilcoxon signed-rank + exact sign test).
    # "ACI better" = diff<0 for gap/Winkler/MPIW, diff>0 for tail_PICP.
    BETTER_IF_NEG = {"gap": True, "Winkler": True, "MPIW": True, "tail_PICP": False}
    mrows = []
    for key in ["gap", "tail_PICP", "Winkler", "MPIW"]:
        cell = []
        for bb in BACKBONES:
            for arm in ARMS:
                d = metric_series(bb, arm, key, "aci") - metric_series(bb, arm, key, "split")
                d = d[np.isfinite(d)]
                if len(d):
                    cell.append(float(d.mean()))   # rep-mean diff for this cell
        d = np.asarray(cell, float)
        n = len(d)
        m, sd = float(d.mean()), float(d.std(ddof=1))
        se = sd / np.sqrt(n)
        tval, pt = stats.ttest_1samp(d, 0.0)
        try:
            _, pw = stats.wilcoxon(d)
        except Exception:
            pw = float("nan")
        better = int(np.sum(d < 0) if BETTER_IF_NEG[key] else np.sum(d > 0))
        p_sign = float(stats.binomtest(better, n, 0.5).pvalue)
        tc = stats.t.ppf(0.975, n - 1)
        mrows.append(dict(metric=key, unit="(backbone,arm)_cell", n_cells=n,
                          mean_aci_minus_split=m, ci95_lo=m - tc * se, ci95_hi=m + tc * se,
                          n_aci_better=better, t=float(tval), p_ttest=float(pt),
                          p_wilcoxon=float(pw), p_sign=p_sign,
                          cohens_dz=m / sd if sd > 0 else float("nan")))
    pd.DataFrame(mrows).to_csv(f"{OUTD}/method_compare.csv", index=False)

    # ---- console summary ----
    print(f"=== Loss-family Friedman (per backbone, reps {reps}) ===")
    for bb in BACKBONES:
        for key, _m in ENDPOINTS:
            r = next(x for x in loss_rows if x["Backbone"] == bb and x["metric"] == key)
            print(f"  {bb:12} {key:13} chi2={r['friedman_chi']:6.2f} p={r['friedman_p']:.4f}")
    print("\n=== TOST equivalence vs Standard (equivalent = CI90 within margin) ===")
    eq = pd.DataFrame(tost_rows)
    for key in ["RMSE", "MPIW_at_0p90", "tail_PICP"]:
        sub = eq[eq.metric == key]
        n_eq = int(sub.equivalent.sum()); n = len(sub)
        print(f"  {key:13}: {n_eq}/{n} arm×backbone contrasts EQUIVALENT")
    print(f"\n=== Method: ACI - split  (unit = (backbone,arm) cell, n={mrows[0]['n_cells']}; "
          f"neg gap/Winkler/MPIW = ACI better) ===")
    for r in mrows:
        sig = "SIG" if (r["ci95_lo"] > 0 or r["ci95_hi"] < 0) else "ns"
        print(f"  {r['metric']:10} d={r['mean_aci_minus_split']:+.3f} "
              f"CI95=[{r['ci95_lo']:+.3f},{r['ci95_hi']:+.3f}] "
              f"ACI_better={r['n_aci_better']}/{r['n_cells']} "
              f"p_w={r['p_wilcoxon']:.3g} p_sign={r['p_sign']:.3g} dz={r['cohens_dz']:+.2f} [{sig}]")
    print(f"\nSaved -> {OUTD}/{{loss_stats,equivalence_tost,method_compare}}.csv")


if __name__ == "__main__":
    main()
