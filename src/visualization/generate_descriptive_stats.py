
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def generate_stats_table():
    # 1. Load Data
    data_path = 'code/data/imputed_daily_data.csv'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return

    # 2. Configure Settings
    # Define variables to analyze (Korean column name -> English label)
    variable_map = {
        '수온 (℃)': 'Water Temperature ($^\circ$C)',
        '수소이온농도': 'pH',
        '전기전도도 (μS/cm)': 'EC ($\mu$S/cm)',
        '용존산소 (mg/L)': 'DO (mg/L)',
        '탁도 (NTU)': 'Turbidity (NTU)',
        '총유기탄소 (mg/L)': 'TOC (mg/L)',
        '총질소 (mg/L)': 'TN (mg/L)',
        '총인 (mg/L)': 'TP (mg/L)',
        '클로로필-a (mg/㎥)': 'Chlorophyll-a (mg/m$^3$)'
    }
    
    site_map = {
        '공주': 'Gongju',
        '대청호': 'Daecheongho',
        '갑천': 'Gapcheon',
        '부여': 'Buyeo',
        '용담호': 'Yongdamho'
    }
    
    # Filter Period (2021-2024 as per user request)
    df['측정일'] = pd.to_datetime(df['측정일'])
    start_date = '2021-01-01'
    end_date = '2024-12-31'
    
    # Filter by date and sites (Although we aggregate across sites, we still filter for validity)
    mask = (df['측정일'] >= start_date) & (df['측정일'] <= end_date) & (df['측정소'].isin(site_map.keys()))
    df_filtered = df.loc[mask].copy()
    
    # 3. Calculate Statistics
    stats_list = []
    
    # Iterate over variables
    for var_kr, var_en in variable_map.items():
        if var_kr not in df_filtered.columns:
            print(f"Warning: Column {var_kr} not found in data.")
            continue
            
        # Aggregate data across ALL stations for this variable
        var_data = df_filtered[var_kr].dropna()
        
        if len(var_data) == 0:
            continue
            
        n = len(var_data)
        mean = var_data.mean()
        median = var_data.median()
        sd = var_data.std()
        min_val = var_data.min()
        max_val = var_data.max()
        cv = sd / mean if mean != 0 else 0
        sk = skew(var_data, bias=False)  # Sample skewness
        ku = kurtosis(var_data, bias=False) # Sample kurtosis (Fisher)
        
        stats_list.append({
            'Variable': var_en,
            'N': n,
            'Mean': mean,
            'Median': median,
            'SD': sd,
            'Min': min_val,
            'Max': max_val,
            'CV': cv,
            'Skewness': sk,
            'Kurtosis': ku
        })
        
    # 4. Generate LaTeX Table
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Descriptive statistics of water quality variables across 5 monitoring stations (2021--2024).}")
    print(r"\label{tab:descriptive_stats}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{lrrrrrrrrr}")
    print(r"\toprule")
    print(r"\textbf{Variable} & \textbf{$N$} & \textbf{Mean} & \textbf{Median} & \textbf{SD} & \textbf{Min} & \textbf{Max} & \textbf{CV} & \textbf{Skewness} & \textbf{Kurtosis} \\")
    print(r"\midrule")
    
    for row in stats_list:
        # Highlight CV > 1.0 logic mentioned in paper
        cv_str = f"{row['CV']:.2f}"
        if row['CV'] > 1.0:
            cv_str = f"\\textbf{{{cv_str}}}"
            
        print(f"{row['Variable']} & {row['N']:,} & {row['Mean']:.2f} & {row['Median']:.2f} & {row['SD']:.2f} & "
              f"{row['Min']:.1f} & {row['Max']:.1f} & {cv_str} & {row['Skewness']:.2f} & {row['Kurtosis']:.2f} \\\\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\\ \footnotesize{\textit{Note:} Aggegated statistics from 5 monitoring stations (Gongju, Daecheongho, Gapcheon, Buyeo, Yongdamho). CV > 1.0 (bold) indicates high variance.}")
    print(r"\end{table*}")

if __name__ == "__main__":
    generate_stats_table()
