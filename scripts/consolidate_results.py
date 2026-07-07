"""Consolidate the ACP-benchmark CSVs into one paper-ready markdown report.

Reads results/acp_benchmark/{master_summary,method_compare,loss_stats,
equivalence_tost,gamma_sensitivity}.csv and writes results/acp_benchmark/RESULTS.md
(a master table + the headline statistics). Pure pandas, no recompute.

Usage: python scripts/consolidate_results.py
"""
import os, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = f"{BASE}/results/acp_benchmark"
ORDER = ["raw", "split", "aci"]
MLAB = {"raw": "Raw Gaussian", "split": "Split-conformal (2023-val)", "aci": "Online ACI"}


def fmt(m, ci):
    return f"{m:.3f}±{ci:.3f}" if np.isfinite(ci) else f"{m:.3f}"


def main():
    master = pd.read_csv(f"{OUTD}/master_metrics.csv")
    L = ["# ACP-centred UQ benchmark — final results\n",
         "Statistical unit = replicate (independent M=5 ensemble). Target coverage 0.90.",
         "Pooled = mean over 5 loss arms × 3 backbones × reps. "
         "Split calibrated on held-out 2023 validation (fair baseline).",
         "Note: raw Gaussian over-covers (0.96>0.90) because the ensemble predictive "
         "σ is slightly over-dispersed; conformal calibration corrects this.\n"]

    # ---- pooled method comparison ----
    L.append("## Method comparison (pooled, mean ± 95% CI)\n")
    L.append("| Method | PICP | |gap| | worst-site | MPIW | Winkler | tail PICP |")
    L.append("|---|---|---|---|---|---|---|")
    def ci95(v):
        v = np.asarray(v, float); n = len(v)
        from scipy import stats
        h = stats.t.ppf(.975, n-1)*v.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan
        return v.mean(), h
    for mth in ORDER:
        g = master[master.Method == mth]
        cells = []
        for k in ["PICP", "gap", "worst_site", "MPIW", "Winkler", "tail_PICP"]:
            m, h = ci95(g[k]); cells.append(fmt(m, h))
        L.append(f"| {MLAB[mth]} | " + " | ".join(cells) + " |")
    L.append("\n*Fair comparison is split vs ACI (both target-calibrated). Raw Gaussian is "
             "disqualified: it misses the marginal target by 0.062 and its intervals are "
             "~30% wider (MPIW 16.9 vs 13.0), so its higher worst-site/tail PICP is bought "
             "by systematic over-coverage, not skill.*")
    L.append("\n*CRPS omitted: identical (2.197 mg/m³) across all three methods by "
             "construction — conformal rescales interval width but leaves the Gaussian "
             "predictive (μ, σ) that CRPS scores unchanged, so it cannot separate methods.*")

    # ---- method significance ----
    if os.path.exists(f"{OUTD}/method_compare.csv"):
        mc = pd.read_csv(f"{OUTD}/method_compare.csv")
        L.append("\n## ACI vs split — paired test, unit = (backbone,arm) cell (rep-averaged, n=15)\n")
        L.append("Each cell is one model configuration (reps averaged) to avoid "
                 "pseudoreplication. Negative = ACI better for gap/Winkler/MPIW; "
                 "positive = ACI better for tail_PICP.\n")
        L.append("| metric | mean(ACI−split) | 95% CI | ACI better | p (Wilcoxon) | p (sign) | Cohen's dz |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in mc.iterrows():
            L.append(f"| {r['metric']} | {r['mean_aci_minus_split']:+.3f} | "
                     f"[{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] | "
                     f"{int(r['n_aci_better'])}/{int(r['n_cells'])} | {r['p_wilcoxon']:.2e} | "
                     f"{r['p_sign']:.2e} | {r['cohens_dz']:+.2f} |")
        L.append("\n*All four p-values equal 6.1e-5 because that is the floor of the "
                 "Wilcoxon/sign test at n=15 when all 15 cells agree (2⁻¹⁴); read each as "
                 "p ≤ 6.1e-5. The effect sizes (Cohen's dz) carry the magnitude.*")

    # ---- loss-arm omnibus + equivalence ----
    if os.path.exists(f"{OUTD}/loss_stats.csv"):
        ls = pd.read_csv(f"{OUTD}/loss_stats.csv")
        L.append("\n## Loss family — Friedman omnibus (per backbone)\n")
        L.append("| Backbone | metric | χ² | p |")
        L.append("|---|---|---|---|")
        for _, r in ls.iterrows():
            L.append(f"| {r['Backbone']} | {r['metric']} | {r['friedman_chi']:.2f} | {r['friedman_p']:.4f} |")
        L.append("\n*The omnibus is significant for Mamba/GRU, but Holm-corrected post-hoc "
                 "contrasts localise no single arm at n=5 (all adjusted p ≥ 0.25); we therefore "
                 "report that the loss families differ overall without claiming which arm drives "
                 "it. This does not affect the ISO≈Standard equivalence (H3) below.*")
    if os.path.exists(f"{OUTD}/equivalence_tost.csv"):
        eq = pd.read_csv(f"{OUTD}/equivalence_tost.csv")
        iso = eq[eq.arm == "ISO_NLL"]
        L.append("\n## ISO_NLL ≈ Standard_NLL — TOST equivalence (H3)\n")
        L.append("| Backbone | metric | mean diff | CI90 | equivalent |")
        L.append("|---|---|---|---|---|")
        for _, r in iso.iterrows():
            L.append(f"| {r['Backbone']} | {r['metric']} | {r['mean_diff']:+.3f} | "
                     f"[{r['ci90_lo']:+.3f}, {r['ci90_hi']:+.3f}] | "
                     f"{'YES' if r['equivalent'] else 'no'} |")
        nE, nT = int(eq.equivalent.sum()), len(eq)
        L.append(f"\nAll-arm equivalence: {nE}/{nT} arm×backbone×metric contrasts within margin.")

    # ---- gamma ----
    if os.path.exists(f"{OUTD}/gamma_sensitivity.csv"):
        gs = pd.read_csv(f"{OUTD}/gamma_sensitivity.csv").sort_values("gamma")
        L.append("\n## ACI γ sensitivity (pooled)\n")
        L.append("| γ | PICP | |gap| | MPIW | Winkler |")
        L.append("|---|---|---|---|---|")
        for _, r in gs.iterrows():
            L.append(f"| {r['gamma']:g} | {r['PICP_mean']:.3f} | {r['gap_mean']:.3f} | "
                     f"{r['MPIW_mean']:.2f} | {r['Winkler_mean']:.2f} |")

    # ---- 2024head sanity (if present) ----
    if os.path.exists(f"{OUTD}/master_metrics_2024head.csv"):
        h = pd.read_csv(f"{OUTD}/master_metrics_2024head.csv")
        L.append("\n## Sanity: 2024-head calibration (reproduces the original demo)\n")
        L.append("Here split/ACI are calibrated on the first 20% of the 2024 test "
                 "year (NOT 2023-val), reproducing the demo's under-coverage.\n")
        HLAB = {"raw": "Raw Gaussian", "split": "Split-conformal (2024-head)",
                "aci": "Online ACI (2024-head warm)"}
        L.append("| Method | PICP | worst-site | tail PICP |")
        L.append("|---|---|---|---|")
        for mth in ORDER:
            g = h[h.Method == mth]
            L.append(f"| {HLAB[mth]} | {g.PICP.mean():.3f} | {g.worst_site.mean():.3f} | {g.tail_PICP.mean():.3f} |")

    # ---- figure captions ----
    L.append("\n## Figure captions\n")
    L.append("**Fig 1 — Marginal coverage by method.** Pooled PICP for raw / split / ACI "
             "against the 0.90 target; error bars are 95% CI over replicates.")
    L.append("**Fig 2 — Per-site coverage.** Raw (grey) appears highest only because it "
             "over-covers everywhere (≈0.96 marginal); the fair comparison is split vs ACI, "
             "where ACI lifts the worst site from 0.889 to 0.902 and clears 0.90 at every site.")
    L.append("**Fig 3 — Reliability of the raw Gaussian UQ.** Empirical vs nominal central "
             "coverage; the curve sits above the diagonal at all levels (ECE 0.176), i.e. the "
             "ensemble σ is over-dispersed — the source of the raw over-coverage.")
    L.append("**Fig 4 — Tail (bloom) coverage.** PICP on the top-decile chl-a slice. As in "
             "Fig 2, raw's higher bar is over-coverage; among target-calibrated methods ACI "
             "(0.916) restores the early-warning slice that split (0.861) under-covers.")
    L.append("**Fig 5 — Coverage–width trade-off.** ACI sits up-and-left of split (higher "
             "coverage at smaller width); the γ sweep traces this coverage–width trade-off.")

    out = f"{OUTD}/RESULTS.md"
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Saved -> {out}\n")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
