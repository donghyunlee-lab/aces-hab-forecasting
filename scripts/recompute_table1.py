"""Recompute Table 1 descriptive statistics for all 19 measured input variables.

Output: LaTeX table rows + sanity comparison with existing Table 1 values.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJ, 'data', 'imputed_daily_data.csv')

SITES = ['공주', '대청호', '갑천', '부여', '용담호']

# Variable specification: (Korean column, English label, units already in column name)
CORE = [
    ('수온 (℃)',           'Water Temperature (\\textcelsius)'),
    ('수소이온농도',         'pH'),
    ('전기전도도 (μS/cm)',  'EC ($\\mu$S/cm)'),
    ('용존산소 (mg/L)',     'DO (mg/L)'),
    ('탁도 (NTU)',          'Turbidity (NTU)'),
    ('총유기탄소 (mg/L)',    'TOC (mg/L)'),
    ('총질소 (mg/L)',       'TN (mg/L)'),
    ('총인 (mg/L)',         'TP (mg/L)'),
    ('클로로필-a (mg/㎥)',  'Chlorophyll-a ($\\mu$g/L)'),
]

VOC = [
    ('염화메틸렌 (μg/L)',                  'Methylene chloride ($\\mu$g/L)'),
    ('1.1.1-트리클로로에테인 (μg/L)',      '1,1,1-Trichloroethane ($\\mu$g/L)'),
    ('사염화탄소 (μg/L)',                  'Carbon tetrachloride ($\\mu$g/L)'),
    ('트리클로로에틸렌 (μg/L)',             'Trichloroethylene ($\\mu$g/L)'),
    ('테트라클로로에틸렌 (μg/L)',           'Tetrachloroethylene ($\\mu$g/L)'),
    ('벤젠 (μg/L)',                        'Benzene ($\\mu$g/L)'),
    ('톨루엔 (μg/L)',                      'Toluene ($\\mu$g/L)'),
    ('에틸벤젠 (μg/L)',                    'Ethylbenzene ($\\mu$g/L)'),
    ('m,p-자일렌 (μg/L)',                  'm,p-Xylene ($\\mu$g/L)'),
    ('o-자일렌 (μg/L)',                    'o-Xylene ($\\mu$g/L)'),
]

EXISTING_TABLE_1 = {
    # English label: (N, Mean, Median, SD, Min, Max, CV, Skew, Kurt)
    'Water Temperature (\\textcelsius)': (7305, 17.647, 18.467, 8.093, 1.167, 32.967, 0.459, -0.121, -1.211),
    'pH': (7305, 7.714, 7.567, 0.838, 6.133, 10.900, 0.109, 0.830, 0.200),
    'EC ($\\mu$S/cm)': (7305, 284.037, 277.333, 151.058, 74.000, 1043.333, 0.532, 0.348, -0.937),
    'DO (mg/L)': (7305, 9.372, 9.433, 2.256, 1.200, 16.133, 0.241, -0.186, -0.251),
    'Turbidity (NTU)': (7305, 6.462, 3.700, 8.781, 0.100, 102.267, 1.359, 3.958, 20.670),
    'TOC (mg/L)': (7305, 3.123, 3.000, 0.949, 1.167, 10.667, 0.304, 0.753, 0.917),
    'TN (mg/L)': (7305, 2.372, 2.062, 1.127, 0.570, 7.396, 0.475, 0.904, 0.141),
    'TP (mg/L)': (7305, 0.030, 0.024, 0.024, 0.003, 0.230, 0.812, 1.933, 6.205),
    'Chlorophyll-a ($\\mu$g/L)': (7305, 19.307, 11.833, 21.328, 0.133, 221.500, 1.105, 2.840, 12.490),
}


def compute_stats(s: pd.Series):
    s = s.dropna()
    n = len(s)
    mean = s.mean()
    median = s.median()
    sd = s.std(ddof=1)
    mn = s.min()
    mx = s.max()
    cv = sd / mean if mean != 0 else float('nan')
    sk = skew(s, bias=False)
    kt = kurtosis(s, fisher=True, bias=False)  # excess kurtosis (subtract 3)
    return n, mean, median, sd, mn, mx, cv, sk, kt


def main():
    df = pd.read_csv(DATA_PATH)
    df['측정일'] = pd.to_datetime(df['측정일'])

    # Filter: 5 sites + 2021-2024
    df = df[df['측정소'].isin(SITES)]
    df = df[(df['측정일'] >= '2021-01-01') & (df['측정일'] <= '2024-12-31')]
    print(f"Filtered rows: {len(df)}")
    print(f"Date range: {df['측정일'].min()} to {df['측정일'].max()}")
    print(f"Sites: {df['측정소'].value_counts().to_dict()}")
    print()

    print("=" * 90)
    print("SANITY CHECK vs existing Table 1 (9 core variables)")
    print("=" * 90)
    mismatches = 0
    for kr, en in CORE:
        if kr not in df.columns:
            print(f"  MISSING COLUMN: {kr}")
            continue
        stats = compute_stats(df[kr])
        if en in EXISTING_TABLE_1:
            old = EXISTING_TABLE_1[en]
            close = all(
                abs(stats[i] - old[i]) < 0.01 * max(1, abs(old[i]))
                for i in range(1, 9)
            ) and stats[0] == old[0]
            status = "OK" if close else "MISMATCH"
            print(f"  {status:8s} {en}")
            print(f"           NEW: N={stats[0]} mean={stats[1]:.3f} median={stats[2]:.3f} sd={stats[3]:.3f} min={stats[4]:.3f} max={stats[5]:.3f} cv={stats[6]:.3f} skew={stats[7]:.3f} kurt={stats[8]:.3f}")
            print(f"           OLD: N={old[0]} mean={old[1]:.3f} median={old[2]:.3f} sd={old[3]:.3f} min={old[4]:.3f} max={old[5]:.3f} cv={old[6]:.3f} skew={old[7]:.3f} kurt={old[8]:.3f}")
            if not close:
                mismatches += 1
        else:
            print(f"  NO BASELINE  {en}: {stats}")

    print(f"\n{mismatches} mismatches in core 9 variables")

    print()
    print("=" * 90)
    print("VOC descriptive statistics (10 new variables)")
    print("=" * 90)
    for kr, en in VOC:
        if kr not in df.columns:
            print(f"  MISSING COLUMN: {kr}")
            continue
        stats = compute_stats(df[kr])
        print(f"  {en}: N={stats[0]} mean={stats[1]:.4f} median={stats[2]:.4f} sd={stats[3]:.4f} min={stats[4]:.4f} max={stats[5]:.4f} cv={stats[6]:.3f} skew={stats[7]:.3f} kurt={stats[8]:.3f}")

    # ------ LaTeX rows ------
    print()
    print("=" * 90)
    print("LaTeX rows (paste into tab_descriptive_stats.tex)")
    print("=" * 90)
    def fmt_row(en, st):
        n = f"{st[0]:,}"
        return (f"{en} & {n} & {st[1]:.3f} & {st[2]:.3f} & {st[3]:.3f} & "
                f"{st[4]:.3f} & {st[5]:.3f} & {st[6]:.3f} & {st[7]:.3f} & {st[8]:.3f} \\\\")
    print("% Core 9 physicochemical variables")
    for kr, en in CORE:
        st = compute_stats(df[kr])
        print(fmt_row(en, st))
    print("\\midrule")
    print("% Trace organic compounds (10)")
    for kr, en in VOC:
        st = compute_stats(df[kr])
        # VOC mostly trace - use more decimals
        n = f"{st[0]:,}"
        print(f"{en} & {n} & {st[1]:.4f} & {st[2]:.4f} & {st[3]:.4f} & "
              f"{st[4]:.4f} & {st[5]:.4f} & {st[6]:.3f} & {st[7]:.3f} & {st[8]:.3f} \\\\")


if __name__ == '__main__':
    main()
