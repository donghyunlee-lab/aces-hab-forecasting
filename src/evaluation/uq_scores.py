"""Additive UQ scoring helpers for the ACP-centred benchmark.

Pure NumPy functions, no dependency on the training pipeline or the
`EvaluationMetrics` class (which the validated pipeline uses) — so importing
this module cannot perturb any "science" path. Everything operates on already
denormalised arrays (original chlorophyll-a scale, mg/m^3).

Provided:
  - winkler_score      : interval score (sharpness + coverage in one number)
  - crps_gaussian      : closed-form CRPS for a Gaussian predictive
  - interval_ece       : mean |nominal - empirical| over central levels
  - matched_mpiw       : MPIW after rescaling intervals so coverage == target
                         (lifted verbatim from scripts/analyze_replicates.py)
  - conditional_coverage: tail / quantile-bin / worst-site coverage diagnostics
"""
import numpy as np
from scipy.stats import norm


def winkler_score(y, lower, upper, alpha):
    """Mean Winkler (interval) score for a central (1-alpha) interval.

    score_i = width_i + (2/alpha) * (distance outside the interval).
    Lower is better; rewards narrow intervals but penalises missed points.
    """
    y = np.asarray(y, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    width = upper - lower
    below = lower - y
    above = y - upper
    penalty = np.where(y < lower, (2.0 / alpha) * below, 0.0) \
        + np.where(y > upper, (2.0 / alpha) * above, 0.0)
    return float(np.mean(width + penalty))


def crps_gaussian(y, mu, sigma):
    """Mean CRPS for a Gaussian predictive N(mu, sigma^2) (closed form).

    CRPS = sigma * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ],  z=(y-mu)/sigma.
    Lower is better. sigma is floored to avoid divide-by-zero.
    """
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-9)
    z = (y - mu) / sigma
    crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0)
                    + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def interval_ece(y, mu, sigma, levels=None):
    """Interval-based expected calibration error.

    For each nominal central coverage c in `levels`, build the symmetric
    Gaussian interval mu +/- z*sigma (z = Phi^{-1}((1+c)/2)), measure empirical
    coverage, and average |c - empirical|. Returns (ece, per_level) where
    per_level is a list of (c, empirical) pairs.
    """
    if levels is None:
        levels = np.arange(0.1, 0.96, 0.05)
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-9)
    gaps, per_level = [], []
    for c in levels:
        z = norm.ppf(0.5 * (1.0 + c))
        emp = float(np.mean((y >= mu - z * sigma) & (y <= mu + z * sigma)))
        gaps.append(abs(c - emp))
        per_level.append((float(c), emp))
    return float(np.mean(gaps)), per_level


def matched_mpiw(y, mu, lo, hi, target=0.90):
    """MPIW after rescaling intervals about mu so empirical coverage == target.

    Identical algorithm to scripts/analyze_replicates.py::matched_mpiw so the
    two analyses report the same coverage-matched width.
    """
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    dl, du = mu - np.asarray(lo, float), np.asarray(hi, float) - mu
    klo, khi = 0.0, 10.0
    for _ in range(40):
        k = 0.5 * (klo + khi)
        cov = np.mean((y >= mu - k * dl) & (y <= mu + k * du))
        if cov < target:
            klo = k
        else:
            khi = k
    k = 0.5 * (klo + khi)
    return float(k * np.mean(dl + du))


def _picp(y, lo, hi):
    return float(np.mean((np.asarray(y, float) >= np.asarray(lo, float))
                         & (np.asarray(y, float) <= np.asarray(hi, float))))


def conditional_coverage(y, lower, upper, site_ids, *, alpha=0.10,
                         bloom_thr=None, tail_q=0.90, n_bins=4):
    """Coverage diagnostics conditioned on target magnitude and site.

    Returns a dict:
      tail_thr        : threshold used for the bloom/tail slice
      tail_picp/_mpiw/_winkler/_n : metrics on y >= threshold (early-warning slice)
      bins            : list of dicts per chlorophyll quantile bin
                        {bin, lo_edge, hi_edge, picp, mpiw, n}
      worst_site_picp : minimum per-site PICP
      site_picp_var   : variance of per-site PICP (coverage stability across sites)
      per_site_picp   : {site_id: picp}

    bloom_thr: absolute chlorophyll-a threshold; if None, the empirical tail_q
    quantile of y is used (default top decile).
    """
    y = np.asarray(y, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    site_ids = np.asarray(site_ids)

    thr = float(np.quantile(y, tail_q)) if bloom_thr is None else float(bloom_thr)
    tail = y >= thr
    out = {"tail_thr": thr, "tail_n": int(tail.sum())}
    if tail.any():
        out["tail_picp"] = _picp(y[tail], lower[tail], upper[tail])
        out["tail_mpiw"] = float(np.mean(upper[tail] - lower[tail]))
        out["tail_winkler"] = winkler_score(y[tail], lower[tail], upper[tail], alpha)
    else:
        out["tail_picp"] = out["tail_mpiw"] = out["tail_winkler"] = float("nan")

    # quantile bins of the target
    edges = np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] = np.inf  # include the max in the last bin
    bins = []
    for b in range(n_bins):
        m = (y >= edges[b]) & (y < edges[b + 1])
        if not m.any():
            continue
        bins.append(dict(bin=b, lo_edge=float(edges[b]),
                         hi_edge=float(edges[b + 1]) if np.isfinite(edges[b + 1]) else float(y.max()),
                         picp=_picp(y[m], lower[m], upper[m]),
                         mpiw=float(np.mean(upper[m] - lower[m])), n=int(m.sum())))
    out["bins"] = bins

    per_site = {}
    for s in np.unique(site_ids):
        m = site_ids == s
        per_site[int(s)] = _picp(y[m], lower[m], upper[m])
    out["per_site_picp"] = per_site
    vals = np.array(list(per_site.values()), float)
    out["worst_site_picp"] = float(vals.min()) if vals.size else float("nan")
    out["site_picp_var"] = float(vals.var(ddof=1)) if vals.size > 1 else float("nan")
    return out
