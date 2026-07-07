"""Operational (decision-layer) evaluation — turns calibrated intervals into
early-warning decisions and scores them. NO retraining: reads the per-step
interval bounds dumped by eval_acp_benchmark.py.

Decision rule (precautionary early warning):
    alert_t = 1{ upper_t >= tau }      upper = raw_hi / split_hi / aci_hi
    event_t = 1{ Actual_t >= tau }     a bloom actually occurred
A precautionary operator alerts whenever the *upper* plausible Chl-a crosses a
bloom threshold tau. A method whose upper bound is well calibrated (neither
over- nor under-covering the bloom tail) gives the best miss/false-alarm trade.

  miss rate        = P(no alert | event)      = FN / n_events     (= bloom-tail
                     upper miscoverage: a real bloom escaped the interval)
  false-alarm rate = P(alert | no event)      = FP / n_nonevents
  FDR              = FP / (FP + TP)            (alerts that were spurious)
  rho-cost         = rho * (#miss) + (#false-alarm), per step, swept over rho
                     (rho = cost(miss)/cost(false-alarm); regulation => rho >> 1)

tau is SWEPT, not fixed: the Korean cyanobacteria-alert thresholds are defined on
cell counts and map only loosely to Chl-a, so a range is more honest than a single
cut. tau is reported in both ug/L and as a percentile of the pooled test Chl-a.
The anchor tau = 90th pct matches the bloom/tail threshold used by the benchmark.

Two regimes (same as the benchmark):
  fair   : 2023-val calibration  (perstep/)            -> the fair baseline
  stress : 2024-head calibration (perstep_2024head/)   -> distribution-shift test

Outputs (results/acp_benchmark/operational/):
  confusion_by_tau.csv  one row per (regime, method, tau): miss/false-alarm/FDR
  rho_cost.csv          one row per (regime, method, rho) at the anchor tau
  lead_time.csv         per-episode alert lead time (representative config)

Usage: python scripts/operational_eval.py
"""
import os, sys, glob
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = f"{BASE}/results/acp_benchmark"
OPD = f"{OUTD}/operational"
METHODS = {"raw": ("raw_lo", "raw_hi"), "split": ("split_lo", "split_hi"),
           "aci": ("aci_lo", "aci_hi")}
REGIMES = {"fair": "perstep", "stress": "perstep_2024head"}
TAU_PCTS = [75, 80, 85, 90, 95, 98]        # tau sweep (percentile of pooled Chl-a)
ANCHOR_PCT = 90                            # headline tau == benchmark bloom/tail thr
RHOS = [1, 2, 3, 5, 10, 20, 50]            # cost(miss)/cost(false-alarm)
# representative single config for the qualitative lead-time / band figure
REP_FILE = "Mamba_Standard_NLL_r0.csv"


def load_regime(subdir):
    """Pool all per-step rows (kept only) across arms x backbones x reps."""
    files = sorted(glob.glob(f"{OUTD}/{subdir}/*.csv"))
    if not files:
        sys.exit(f"no per-step files in {OUTD}/{subdir} — run eval_acp_benchmark.py first")
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d = d[d["kept"] == True]           # noqa: E712 — evaluation slice only
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def confusion(actual, upper, tau):
    event = actual >= tau
    alert = upper >= tau
    n_ev = int(event.sum()); n_nev = int((~event).sum())
    tp = int((event & alert).sum()); fn = int((event & ~alert).sum())
    fp = int((~event & alert).sum()); tn = int((~event & ~alert).sum())
    return dict(
        n=len(actual), n_events=n_ev, n_nonevents=n_nev,
        TP=tp, FN=fn, FP=fp, TN=tn,
        miss_rate=(fn / n_ev) if n_ev else np.nan,
        false_alarm_rate=(fp / n_nev) if n_nev else np.nan,
        FDR=(fp / (fp + tp)) if (fp + tp) else np.nan,
    )


def episodes_lead_time(df, hi_col, tau, max_lead=14):
    """Mean alert lead time (days) over bloom episodes, on a sorted single config.

    For each site, a bloom episode = a maximal run of days with Actual>=tau.
    Lead = days between the episode onset and the FIRST alert (upper>=tau) within
    the `max_lead` days preceding onset (0 if alert only fires at onset; NaN if no
    alert in the window => effectively a miss for warning purposes).
    """
    leads = []
    for s, g in df.sort_values(["Station_ID", "Date"]).groupby("Station_ID"):
        a = g["Actual"].to_numpy(float) >= tau
        u = g[hi_col].to_numpy(float) >= tau
        onsets = np.where(a & ~np.r_[False, a[:-1]])[0]    # rising edges of events
        for o in onsets:
            w0 = max(0, o - max_lead)
            pre = np.where(u[w0:o + 1])[0]                 # alert indices in window
            leads.append((o - (w0 + pre[0])) if len(pre) else np.nan)
    leads = np.array(leads, float)
    return dict(n_episodes=len(leads),
                warned_frac=float(np.mean(~np.isnan(leads))) if len(leads) else np.nan,
                mean_lead=float(np.nanmean(leads)) if np.any(~np.isnan(leads)) else np.nan,
                median_lead=float(np.nanmedian(leads)) if np.any(~np.isnan(leads)) else np.nan)


def main():
    os.makedirs(OPD, exist_ok=True)
    pooled = {rg: load_regime(sub) for rg, sub in REGIMES.items()}

    # tau grid from the FAIR pooled Chl-a so both regimes share comparable cuts
    base_actual = pooled["fair"]["Actual"].to_numpy(float)
    taus = {p: float(np.percentile(base_actual, p)) for p in TAU_PCTS}
    print(f"Chl-a (pooled fair test): min {base_actual.min():.1f}  "
          f"median {np.median(base_actual):.1f}  max {base_actual.max():.1f} (ug/L)")
    print("tau grid (ug/L): " + ", ".join(f"p{p}={t:.1f}" for p, t in taus.items()))
    anchor_tau = taus[ANCHOR_PCT]

    # ---- confusion by tau ----
    rows = []
    for rg, df in pooled.items():
        actual = df["Actual"].to_numpy(float)
        for mth, (_, hi) in METHODS.items():
            up = df[hi].to_numpy(float)
            for p, t in taus.items():
                c = confusion(actual, up, t)
                rows.append(dict(regime=rg, method=mth, tau_pct=p,
                                 tau_ugL=round(t, 2), **c))
    conf = pd.DataFrame(rows)
    conf.to_csv(f"{OPD}/confusion_by_tau.csv", index=False)

    # ---- rho-cost at anchor tau ----
    crows = []
    for rg, df in pooled.items():
        actual = df["Actual"].to_numpy(float)
        n = len(actual)
        for mth, (_, hi) in METHODS.items():
            c = confusion(actual, df[hi].to_numpy(float), anchor_tau)
            for rho in RHOS:
                crows.append(dict(regime=rg, method=mth, tau_pct=ANCHOR_PCT,
                                  tau_ugL=round(anchor_tau, 2), rho=rho,
                                  exp_cost_per_1000=1000.0 * (rho * c["FN"] + c["FP"]) / n,
                                  miss_rate=c["miss_rate"],
                                  false_alarm_rate=c["false_alarm_rate"]))
    pd.DataFrame(crows).to_csv(f"{OPD}/rho_cost.csv", index=False)

    # ---- lead time (representative single config) ----
    lrows = []
    for rg, sub in REGIMES.items():
        fp = f"{OUTD}/{sub}/{REP_FILE}"
        if not os.path.exists(fp):
            continue
        d = pd.read_csv(fp); d = d[d["kept"] == True]    # noqa: E712
        for mth, (_, hi) in METHODS.items():
            lt = episodes_lead_time(d, hi, anchor_tau)
            lrows.append(dict(regime=rg, method=mth, config=REP_FILE,
                              tau_ugL=round(anchor_tau, 2), **lt))
    pd.DataFrame(lrows).to_csv(f"{OPD}/lead_time.csv", index=False)

    # ---- console headline ----
    print(f"\n=== Operational confusion @ anchor tau = p{ANCHOR_PCT} "
          f"({anchor_tau:.1f} ug/L) — pooled over arms x backbones x reps ===")
    hdr = f"{'regime':7} {'method':6} | {'miss%':>7} {'falarm%':>8} {'FDR%':>7}"
    print(hdr); print("-" * len(hdr))
    for rg in REGIMES:
        for mth in METHODS:
            r = conf[(conf.regime == rg) & (conf.method == mth) &
                     (conf.tau_pct == ANCHOR_PCT)].iloc[0]
            print(f"{rg:7} {mth:6} | {100*r.miss_rate:7.1f} "
                  f"{100*r.false_alarm_rate:8.1f} {100*r.FDR:7.1f}")
    print(f"\nSaved -> {OPD}/{{confusion_by_tau,rho_cost,lead_time}}.csv")


if __name__ == "__main__":
    main()
