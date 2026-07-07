"""SHAP feature-attribution analysis — Mamba + ISO-NLL + 5-member ensemble.

Stages controlled by --stage:
  sanity   : load preprocessor + ensemble, compare with raw_preds CSV, run SHAP on 1 sample
  compute  : full per-member SHAP across 5 members; aggregate global / sigma-stratified / site
  plots    : generate supplementary figures + tables
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from src.data.preprocessor import DataPreprocessor
from src.models.builder import get_model
from src.evaluation.shap import MetaSHAP
from scripts.run_experiment import create_config

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                      'cuda' if torch.cuda.is_available() else 'cpu')

OUT_DIR = os.path.join(PROJ, 'results', '2026-02-07', 'revision_analysis', 'shap')
RAW_DIR = os.path.join(OUT_DIR, 'raw')
PLOTS_DIR = os.path.join(OUT_DIR, 'plots')
for d in [OUT_DIR, RAW_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

ENSEMBLE_PATHS = [os.path.join(PROJ, 'models', f'Mamba_Ensemble_Member{m}.pth') for m in range(5)]
ENSEMBLE_PRED_CSV = os.path.join(PROJ, 'results', '2026-02-07', 'raw_preds', 'phase2',
                                 'Ensemble_ADNLL_s1_preds.csv')
SITE_NAMES = ['공주', '대청호', '갑천', '부여', '용담호']
SITE_NAMES_EN = ['Gongju', 'Daecheongho', 'Gapcheon', 'Buyeo', 'Yongdamho']


def load_data():
    base_cfg = create_config(model_type='Mamba', uq_method='Ensemble')
    base_cfg['decoupled'] = True
    pre = DataPreprocessor(
        data_path=base_cfg['data_path'],
        site_names=base_cfg['site_names'],
        apply_log_transform=base_cfg.get('apply_log_transform', True),
        use_panel_data=True,
    )
    df = pre.load_data()
    df_norm, feats = pre.preprocess(df)
    split = pre.split_train_test(
        df, df_norm,
        seq_len=base_cfg['seq_len'],
        train_period=base_cfg['train_period'],
        val_period=base_cfg['val_period'],
        test_period=base_cfg['test_period'],
    )
    return base_cfg, pre, feats, split


def build_member(cfg, n_features, n_stations, ckpt_path):
    model = get_model(cfg, n_features=n_features, n_stations=n_stations).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def ensemble_forward(models, X, sids, batch_size=256):
    """Mean of (mean, var) across ensemble members. Returns numpy arrays (N, 1)."""
    means, vars_ = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEVICE)
            sb = sids[i:i+batch_size].to(DEVICE) if sids is not None else None
            mu_list, va_list = [], []
            for m in models:
                mu, va = m(xb, sb)
                mu_list.append(mu)
                va_list.append(va)
            mu_stack = torch.stack(mu_list, 0)   # (M, B, 1)
            va_stack = torch.stack(va_list, 0)   # (M, B, 1)
            mu_mean = mu_stack.mean(0)
            # ensemble var: aleatoric (mean of vars) + epistemic (var of means)
            va_mean = va_stack.mean(0) + mu_stack.var(0, unbiased=False)
            means.append(mu_mean.cpu().numpy())
            vars_.append(va_mean.cpu().numpy())
    return np.concatenate(means, 0), np.concatenate(vars_, 0)


def denormalize(pred_mean_norm, pred_var_norm, pre, n_features):
    target_idx = pre.target_idx
    data_min = pre.scaler.data_min_[target_idx]
    data_max = pre.scaler.data_max_[target_idx]
    data_range = data_max - data_min
    pm = pred_mean_norm.squeeze(-1) if pred_mean_norm.ndim > 1 else pred_mean_norm
    var_d = pred_var_norm.squeeze(-1) * (data_range ** 2)
    dummy = np.zeros((len(pm), n_features))
    dummy[:, target_idx] = pm
    pm_d = pre.inverse_transform(dummy)[:, target_idx]
    if pre.apply_log_transform:
        var_d = var_d * ((pm_d + 1) ** 2)
    return pm_d, var_d


def stratified_subsample(station_ids, per_site=200, seed=0):
    rng = np.random.default_rng(seed)
    idx_all = []
    for sid in range(5):
        idx = np.where(station_ids == sid)[0]
        if len(idx) <= per_site:
            chosen = idx
        else:
            chosen = rng.choice(idx, size=per_site, replace=False)
        idx_all.append(chosen)
    return np.sort(np.concatenate(idx_all))


# ---------------------------------------------------------------------------
# Stage 1: sanity
# ---------------------------------------------------------------------------
def run_sanity():
    print(f"[sanity] device = {DEVICE}")
    cfg, pre, feats, split = load_data()
    n_features = len(feats)
    n_stations = len(SITE_NAMES)
    print(f"[sanity] n_features={n_features}  n_test={split['X_test'].shape[0]}")

    # ---- 1. Load 5 members
    for p in ENSEMBLE_PATHS:
        if not os.path.exists(p):
            print(f"[sanity] MISSING checkpoint {p}")
            return False
    cfg_member = dict(cfg)
    cfg_member['decoupled'] = True
    members = [build_member(cfg_member, n_features, n_stations, p) for p in ENSEMBLE_PATHS]
    print(f"[sanity] loaded {len(members)} ensemble members")

    # ---- 2. Forward on test, compare to saved CSV
    X_test = torch.from_numpy(split['X_test']).float()
    sids = torch.from_numpy(split['test_station_ids']).long()
    pm_n, pv_n = ensemble_forward(members, X_test, sids)
    pm_d, pv_d = denormalize(pm_n, pv_n, pre, n_features)
    sigma = np.sqrt(pv_d)

    # Compare with saved Ensemble_ADNLL_s1_preds.csv
    if os.path.exists(ENSEMBLE_PRED_CSV):
        ref = pd.read_csv(ENSEMBLE_PRED_CSV)
        n = min(len(ref), len(pm_d))
        ref_pm = ref['Predicted_Mean'].values[:n]
        ref_st = ref['Predicted_Std'].values[:n]
        pm_corr = np.corrcoef(pm_d[:n], ref_pm)[0, 1]
        st_corr = np.corrcoef(sigma[:n], ref_st)[0, 1]
        rmse_pm = float(np.sqrt(np.mean((pm_d[:n] - ref_pm) ** 2)))
        print(f"[sanity] vs {os.path.basename(ENSEMBLE_PRED_CSV)} (N={n}):")
        print(f"           Predicted_Mean: corr={pm_corr:.4f}  RMSE={rmse_pm:.3f}")
        print(f"           Predicted_Std : corr={st_corr:.4f}")
        # sample check
        print(f"           our  mu[:3] = {pm_d[:3]}")
        print(f"           ref  mu[:3] = {ref_pm[:3]}")
    else:
        print(f"[sanity] reference CSV not found, skipping comparison")

    # ---- 3. SHAP single-sample sanity on member 0
    model = members[0]
    feat_names = feats
    bg_idx = stratified_subsample(split['test_station_ids'], per_site=10, seed=42)
    X_bg = split['X_test'][bg_idx[:30]]
    sid_bg = split['test_station_ids'][bg_idx[:30]]
    X_one = split['X_test'][bg_idx[:5]]
    sid_one = split['test_station_ids'][bg_idx[:5]]

    ms = MetaSHAP(model, feat_names, X_background=X_bg, sid_background=sid_bg)
    def target_pred(m, v): return m
    t0 = time.time()
    shap_vals = ms.compute_all_shap_values(X_one, target_pred, sid_one)
    dt = time.time() - t0
    print(f"[sanity] SHAP shape={shap_vals.shape}  time={dt:.2f}s for 5 samples / 30 bg")
    print(f"[sanity] |SHAP| mean per feature: {np.mean(np.abs(shap_vals), axis=(0, 1))}")

    # ---- 4. estimate full-run time
    n_full = 5 * 5 * 200  # 5 members × 5 sites × 200 samples = 5000
    bg_full = 50
    # Approximation: time scales linearly in samples × bg
    est_per_member = dt * (1000 / 5) * (bg_full / 30)
    est_total = est_per_member * 5
    print(f"[sanity] estimated full SHAP time: ~{est_per_member/60:.1f}min/member, ~{est_total/60:.1f}min total")

    return True


# ---------------------------------------------------------------------------
# Stage 2: full computation
# ---------------------------------------------------------------------------
def run_compute(per_site=200, n_bg=50, seed=42):
    cfg, pre, feats, split = load_data()
    n_features = len(feats)
    n_stations = len(SITE_NAMES)
    cfg['decoupled'] = True

    members = [build_member(cfg, n_features, n_stations, p) for p in ENSEMBLE_PATHS]

    # ---- ensemble σ for stratification (use ensemble of all 5 members)
    X_test = torch.from_numpy(split['X_test']).float()
    sids_t = torch.from_numpy(split['test_station_ids']).long()
    pm_n, pv_n = ensemble_forward(members, X_test, sids_t)
    pm_d, pv_d = denormalize(pm_n, pv_n, pre, n_features)
    sigma = np.sqrt(pv_d)

    # ---- subsample (same indices across all members)
    sample_idx = stratified_subsample(split['test_station_ids'], per_site=per_site, seed=seed)
    X_sub = split['X_test'][sample_idx]
    sid_sub = split['test_station_ids'][sample_idx]
    sigma_sub = sigma[sample_idx]

    # high/low-σ thresholds (within subsample)
    hi_thr = np.quantile(sigma_sub, 0.75)
    lo_thr = np.quantile(sigma_sub, 0.25)
    hi_idx_in_sub = np.where(sigma_sub >= hi_thr)[0]
    lo_idx_in_sub = np.where(sigma_sub <= lo_thr)[0]
    print(f"[compute] subsample N={len(sample_idx)}  high-σ={len(hi_idx_in_sub)}  low-σ={len(lo_idx_in_sub)}")

    # background: random 50 from training (first try test)
    rng = np.random.default_rng(seed)
    bg_pool_idx = rng.choice(len(split['X_train']), size=n_bg, replace=False)
    X_bg = split['X_train'][bg_pool_idx]
    sid_bg = split['train_station_ids'][bg_pool_idx]

    def target_pred(m, v): return m
    all_shap = []  # list of (N_sub, seq, feat) per member
    for m_idx, model in enumerate(members):
        ms = MetaSHAP(model, feats, X_background=X_bg, sid_background=sid_bg)
        t0 = time.time()
        sv = ms.compute_all_shap_values(X_sub, target_pred, sid_sub)
        if sv.ndim == 4 and sv.shape[-1] == 1:
            sv = sv.squeeze(-1)
        np.save(os.path.join(RAW_DIR, f'shap_values_m{m_idx}.npy'), sv)
        all_shap.append(sv)
        print(f"[compute] member {m_idx} done shape={sv.shape} time={time.time()-t0:.1f}s")

    all_shap = np.stack(all_shap, axis=0)  # (M, N, seq, feat)
    np.save(os.path.join(RAW_DIR, 'shap_values_all.npy'), all_shap)
    np.save(os.path.join(OUT_DIR, 'sample_idx.npy'), sample_idx)
    np.save(os.path.join(OUT_DIR, 'sigma_sub.npy'), sigma_sub)
    np.save(os.path.join(OUT_DIR, 'hi_idx_in_sub.npy'), hi_idx_in_sub)
    np.save(os.path.join(OUT_DIR, 'lo_idx_in_sub.npy'), lo_idx_in_sub)
    np.save(os.path.join(OUT_DIR, 'sid_sub.npy'), sid_sub)
    np.save(os.path.join(OUT_DIR, 'X_sub.npy'), X_sub)
    with open(os.path.join(OUT_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feats, f, ensure_ascii=False)

    # ---- aggregations
    abs_shap = np.abs(all_shap)  # (M, N, seq, feat)

    # Global: mean over (seq, samples) per member, then mean/std across members
    glob_per_member = abs_shap.mean(axis=(1, 2))  # (M, feat)
    glob_mean = glob_per_member.mean(0)
    glob_std = glob_per_member.std(0)
    pd.DataFrame({'Feature': feats, 'mean_abs_shap': glob_mean, 'std_abs_shap': glob_std}) \
        .to_csv(os.path.join(OUT_DIR, 'importance_global.csv'), index=False)

    # σ-stratified
    rows = []
    for grp_name, idx in [('high_sigma', hi_idx_in_sub), ('low_sigma', lo_idx_in_sub)]:
        grp_per_member = abs_shap[:, idx].mean(axis=(1, 2))  # (M, feat)
        rows.append(pd.DataFrame({
            'group': grp_name, 'Feature': feats,
            'mean_abs_shap': grp_per_member.mean(0),
            'std_abs_shap': grp_per_member.std(0),
        }))
    pd.concat(rows).to_csv(os.path.join(OUT_DIR, 'importance_by_sigma_group.csv'), index=False)

    # Site-level
    site_rows = []
    for sid in range(5):
        site_in_sub = np.where(sid_sub == sid)[0]
        if len(site_in_sub) == 0:
            continue
        per_m = abs_shap[:, site_in_sub].mean(axis=(1, 2))  # (M, feat)
        site_rows.append(pd.DataFrame({
            'site': SITE_NAMES[sid], 'Feature': feats,
            'mean_abs_shap': per_m.mean(0), 'std_abs_shap': per_m.std(0),
        }))
    pd.concat(site_rows).to_csv(os.path.join(OUT_DIR, 'importance_by_site.csv'), index=False)

    # Site × σ-group
    sxs = []
    for sid in range(5):
        for grp_name, idx in [('high_sigma', hi_idx_in_sub), ('low_sigma', lo_idx_in_sub)]:
            site_in = np.where(sid_sub == sid)[0]
            both = np.intersect1d(site_in, idx)
            if len(both) == 0:
                continue
            per_m = abs_shap[:, both].mean(axis=(1, 2))
            sxs.append(pd.DataFrame({
                'site': SITE_NAMES[sid], 'group': grp_name, 'Feature': feats,
                'n': len(both),
                'mean_abs_shap': per_m.mean(0), 'std_abs_shap': per_m.std(0),
            }))
    pd.concat(sxs).to_csv(os.path.join(OUT_DIR, 'importance_by_site_sigma.csv'), index=False)

    print('[compute] all aggregates saved to', OUT_DIR)


# ---------------------------------------------------------------------------
# Stage 3: plots & tables
# ---------------------------------------------------------------------------
def run_plots():
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

    feats = json.load(open(os.path.join(OUT_DIR, 'feature_names.json')))
    all_shap = np.load(os.path.join(RAW_DIR, 'shap_values_all.npy'))  # (M, N, seq, feat)
    X_sub = np.load(os.path.join(OUT_DIR, 'X_sub.npy'))
    hi_idx = np.load(os.path.join(OUT_DIR, 'hi_idx_in_sub.npy'))
    lo_idx = np.load(os.path.join(OUT_DIR, 'lo_idx_in_sub.npy'))
    sid_sub = np.load(os.path.join(OUT_DIR, 'sid_sub.npy'))

    # English labels for figures
    label_map = {
        '수온 (℃)': 'Water Temp.',
        '수소이온농도': 'pH',
        '전기전도도 (μS/cm)': 'EC',
        '용존산소 (mg/L)': 'DO',
        '탁도 (NTU)': 'Turbidity',
        '총유기탄소 (mg/L)': 'TOC',
        '총질소 (mg/L)': 'TN',
        '총인 (mg/L)': 'TP',
        '클로로필-a (mg/㎥)': 'Chl-a',
        '클로로필-a (mg/㎥)_diff1': 'Chl-a d1d',
        '클로로필-a (mg/㎥)_ma7': 'Chl-a MA7',
        '염화메틸렌 (μg/L)': 'Methylene chloride',
        '1.1.1-트리클로로에테인 (μg/L)': '1,1,1-TCA',
        '사염화탄소 (μg/L)': 'CCl4',
        '트리클로로에틸렌 (μg/L)': 'TCE',
        '테트라클로로에틸렌 (μg/L)': 'PCE',
        '벤젠 (μg/L)': 'Benzene',
        '톨루엔 (μg/L)': 'Toluene',
        '에틸벤젠 (μg/L)': 'Ethylbenzene',
        'm,p-자일렌 (μg/L)': 'm,p-Xylene',
        'o-자일렌 (μg/L)': 'o-Xylene',
    }
    feat_en = [label_map.get(f, f) for f in feats]

    # ---- S-Fig X1: Beeswarm (member 0, prediction μ)
    sv0 = all_shap[0]
    last_shap = sv0[:, -1, :]
    last_X = X_sub[:, -1, :]
    try:
        import shap as shap_lib
        plt.figure(figsize=(8, 5))
        shap_lib.summary_plot(last_shap, last_X, feature_names=feat_en, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'beeswarm_global_mu.png'), dpi=200, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print('[plots] beeswarm failed:', e)

    # ---- S-Fig X2: high vs low σ bar
    abs_shap = np.abs(all_shap)
    hi_per_m = abs_shap[:, hi_idx].mean(axis=(1, 2))  # (M, F)
    lo_per_m = abs_shap[:, lo_idx].mean(axis=(1, 2))
    hi_mean, hi_std = hi_per_m.mean(0), hi_per_m.std(0)
    lo_mean, lo_std = lo_per_m.mean(0), lo_per_m.std(0)

    order = np.argsort(-hi_mean)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(feats))
    w = 0.4
    ax.bar(x - w/2, hi_mean[order], w, yerr=hi_std[order],
           label='High σ (top 25%)', color='#d6604d', capsize=3)
    ax.bar(x + w/2, lo_mean[order], w, yerr=lo_std[order],
           label='Low σ (bottom 25%)', color='#4393c3', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([feat_en[i] for i in order], rotation=35, ha='right')
    ax.set_ylabel('Mean |SHAP| (5-member mean ± std)')
    ax.set_title('Feature attribution under high vs low predicted uncertainty')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'bar_high_vs_low_sigma.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ---- S-Fig X3: site × feature heatmap (5-member mean)
    site_mat = np.zeros((5, len(feats)))
    for sid in range(5):
        site_in = np.where(sid_sub == sid)[0]
        if len(site_in) == 0:
            continue
        site_mat[sid] = abs_shap[:, site_in].mean(axis=(0, 1, 2))
    # row-normalise for visual clarity
    site_norm = site_mat / (site_mat.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(site_norm, annot=site_mat, fmt='.4f', cmap='viridis',
                xticklabels=feat_en, yticklabels=SITE_NAMES_EN, ax=ax,
                annot_kws={'size': 8},
                cbar_kws={'label': 'Row-normalised |SHAP|'})
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    ax.set_title('Site-specific feature importance (5-member ensemble mean)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'heatmap_site_feature.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ---- S-Tab X1: top-5 by site (overall / high / low)
    rows = []
    for sid in range(5):
        for grp_name, idx in [('overall', np.arange(all_shap.shape[1])),
                              ('high_sigma', hi_idx),
                              ('low_sigma', lo_idx)]:
            site_in = np.where(sid_sub == sid)[0]
            both = np.intersect1d(site_in, idx)
            if len(both) == 0:
                continue
            per_m = abs_shap[:, both].mean(axis=(1, 2))  # (M, feat)
            mean_imp = per_m.mean(0)
            std_imp = per_m.std(0)
            top5 = np.argsort(-mean_imp)[:5]
            for rank, fi in enumerate(top5, 1):
                rows.append({
                    'Site': SITE_NAMES_EN[sid],
                    'Group': grp_name,
                    'Rank': rank,
                    'Feature': feat_en[fi],
                    'Mean_|SHAP|': f"{mean_imp[fi]:.5f}",
                    'Std_|SHAP|': f"{std_imp[fi]:.5f}",
                    'n_samples': len(both),
                })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'table_top5_by_site_group.csv'), index=False)

    # ---- S-Tab X2: 9 × 5 importance matrix (mean ± std)
    glob_per_m = abs_shap.mean(axis=(1, 2))  # (M, feat)
    glob_mean, glob_std = glob_per_m.mean(0), glob_per_m.std(0)
    cols = {'Feature': feat_en}
    for sid in range(5):
        site_in = np.where(sid_sub == sid)[0]
        per_m = abs_shap[:, site_in].mean(axis=(1, 2)) if len(site_in) > 0 else np.zeros((all_shap.shape[0], len(feats)))
        cols[f'{SITE_NAMES_EN[sid]}_mean'] = per_m.mean(0)
        cols[f'{SITE_NAMES_EN[sid]}_std'] = per_m.std(0)
    cols['Global_mean'] = glob_mean
    cols['Global_std'] = glob_std
    pd.DataFrame(cols).to_csv(os.path.join(OUT_DIR, 'table_importance_matrix.csv'), index=False)
    print('[plots] all figures + tables saved to', PLOTS_DIR, OUT_DIR)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['sanity', 'compute', 'plots'], required=True)
    ap.add_argument('--per-site', type=int, default=200)
    ap.add_argument('--n-bg', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.stage == 'sanity':
        ok = run_sanity()
        sys.exit(0 if ok else 1)
    elif args.stage == 'compute':
        run_compute(per_site=args.per_site, n_bg=args.n_bg, seed=args.seed)
    elif args.stage == 'plots':
        run_plots()
