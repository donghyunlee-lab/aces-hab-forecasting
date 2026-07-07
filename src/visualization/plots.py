
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from scipy import stats
from sklearn.calibration import calibration_curve

def plot_beeswarm_custom(df_beeswarm, save_path):
    """
    Generate SHAP beeswarm plot using seaborn stripplot.
    """
    plt.figure(figsize=(10, 12))
    
    # Validate required columns
    required_cols = ['Feature', 'SHAP_Value']
    for col in required_cols:
        if col not in df_beeswarm.columns:
            raise ValueError(f"Required column '{col}' not found in df_beeswarm. Available columns: {df_beeswarm.columns.tolist()}")
    
    # Sort features by mean absolute SHAP value
    feature_importance = df_beeswarm.groupby('Feature')['SHAP_Value'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
    sorted_features = feature_importance.index[:20]  # Top 20
    
    # Filter data
    plot_data = df_beeswarm[df_beeswarm['Feature'].isin(sorted_features)].copy()
    
    # Check if 'Value' column exists for hue coloring
    if 'Value' in plot_data.columns:
        hue_col = 'Value'
    elif 'Feature_Value' in plot_data.columns:
        hue_col = 'Feature_Value'
    else:
        # Create dummy hue column based on SHAP values
        hue_col = None
    
    # Create plot with seaborn compatibility
    if hue_col is not None:
        sns.stripplot(data=plot_data, x='SHAP_Value', y='Feature', hue=hue_col, 
                      palette='coolwarm', alpha=0.5, size=2, order=sorted_features,
                      dodge=False, legend=False)
    else:
        sns.stripplot(data=plot_data, x='SHAP_Value', y='Feature',
                      color='steelblue', alpha=0.5, size=2, order=sorted_features,
                      dodge=False)
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.2)
    plt.title('SHAP Value (Impact on Model Output)')
    plt.xlabel('SHAP Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

class HABPublicationVisualizer:
    """
    KDD ADS 트랙 및 고수준 저널 투고를 위한 통합 시각화 모듈.
    성능(Performance), 신뢰성(Reliability), 설명 가능성(Explainability) 3대 테마로 구성.
    """
    def __init__(self, save_dir='results/visualizations/publication'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # 한글 폰트 설정 (필요 시) 및 스타일 설정
        plt.style.use('seaborn-v0_8-whitegrid')
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

    def plot_fig1_performance_suite(self, results_df):
        """Figure 1: Predictive Performance (Time-series & Multi-site Scatter)"""
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        # 1.1 Representative Site Time-series (Gongju or Best site)
        ax1 = fig.add_subplot(gs[0, :])
        site_name = results_df['Site'].iloc[0] if 'Site' in results_df.columns else 'Global'
        # Default to Gongju or first site found
        if 'Site' in results_df.columns and '공주' in results_df['Site'].values:
            site_name = '공주'
            
        site_df = results_df[results_df['Site'] == site_name].sort_values('Date').tail(180) # 최근 6개월
        
        ax1.plot(pd.to_datetime(site_df['Date']), site_df['Actual'], 'k-o', label='Observation', markersize=3, alpha=0.7)
        ax1.plot(pd.to_datetime(site_df['Date']), site_df['Predicted_Mean'], 'b-', label='Prediction (iTransformer)', linewidth=2)
        ax1.fill_between(pd.to_datetime(site_df['Date']), site_df['CI_Lower'], site_df['CI_Upper'], color='blue', alpha=0.2, label='95% Calibrated CI')
        ax1.set_title(f'Figure 1(a): Forecasting Performance with Calibrated Confidence Intervals ({site_name})', fontsize=15, fontweight='bold')
        ax1.legend(loc='upper right', frameon=True)
        ax1.set_ylabel('Chl-a (mg/m³)')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.grid(True, linestyle='--', alpha=0.5)

        # 1.2 Multi-site Integration Scatter
        ax2 = fig.add_subplot(gs[1, 0])
        if 'Site' in results_df.columns:
            sns.scatterplot(data=results_df, x='Actual', y='Predicted_Mean', hue='Site', alpha=0.5, ax=ax2, palette='viridis')
        else:
            sns.scatterplot(data=results_df, x='Actual', y='Predicted_Mean', alpha=0.5, ax=ax2, color='blue')
            
        max_val = max(results_df['Actual'].max(), results_df['Predicted_Mean'].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', label='Ideal (1:1)')
        ax2.set_title('Figure 1(b): Actual vs. Predicted (All Sites)', fontsize=13)
        ax2.set_xlabel('Measured Chl-a')
        ax2.set_ylabel('Estimated Chl-a')
        ax2.grid(True, linestyle='--', alpha=0.5)

        # 1.3 Site-wise RMSE/R2 Comparison (Bar)
        ax3 = fig.add_subplot(gs[1, 1])
        if 'Site' in results_df.columns:
            site_metrics = results_df.groupby('Site').apply(lambda x: np.sqrt(((x['Actual'] - x['Predicted_Mean'])**2).mean())).reset_index(name='RMSE')
            sns.barplot(data=site_metrics, x='Site', y='RMSE', palette='magma', ax=ax3)
            ax3.set_title('Figure 1(c): Spatial Robustness (Site-wise RMSE)', fontsize=13)
        else:
            ax3.text(0.5, 0.5, "Single Site Mode", ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/Fig1_Performance_Suite.png', dpi=300)
        plt.close()

    def plot_fig2_reliability_suite(self, results_df):
        """Figure 2: UQ Reliability (Reliability Diagram & Error-Uncertainty Correlation)"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 2.1 Reliability Diagram (Calibration Curve)
        ax1 = axes[0]
        # Check if Predicted_Std exists, if not infer from CI_Upper/Lower (approx 1.96 std)
        if 'Predicted_Std' not in results_df.columns:
             results_df['Predicted_Std'] = (results_df['CI_Upper'] - results_df['Predicted_Mean']) / 1.96

        # Binning by predicted interval width or variance
        results_df['Error'] = np.abs(results_df['Actual'] - results_df['Predicted_Mean'])
        
        # PICP check across different confidence levels (simplified)
        confidence_levels = np.linspace(0.1, 0.95, 10)
        actual_coverage = []
        for cl in confidence_levels:
            z = stats.norm.ppf(1 - (1-cl)/2)
            lower = results_df['Predicted_Mean'] - z * results_df['Predicted_Std']
            upper = results_df['Predicted_Mean'] + z * results_df['Predicted_Std']
            coverage = ((results_df['Actual'] >= lower) & (results_df['Actual'] <= upper)).mean()
            actual_coverage.append(coverage)
            
        ax1.plot(confidence_levels, actual_coverage, 's-', color='darkblue', label='Proposed (Laplace+ACP)')
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.set_title('Figure 2(a): Reliability Diagram', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Expected Confidence Level')
        ax1.set_ylabel('Observed Coverage (PICP)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # 2.2 Error vs. Uncertainty Correlation (Is the model "honest" about its error?)
        ax2 = axes[1]
        sns.regplot(data=results_df, x='Predicted_Std', y='Error', 
                    scatter_kws={'alpha':0.3, 'color':'gray', 's': 10}, line_kws={'color':'red'}, ax=ax2)
        corr, _ = stats.pearsonr(results_df['Predicted_Std'], results_df['Error'])
        ax2.set_title(f'Figure 2(b): Error-Uncertainty Correlation (r={corr:.3f})', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Uncertainty (Std Dev)')
        ax2.set_ylabel('Absolute Prediction Error')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/Fig2_Reliability_Suite.png', dpi=300)
        plt.close()

    def plot_fig3_global_explainability(self, beeswarm_df, site_wise_df):
        """Figure 3: Global Interpretability (Beeswarm & Site Heatmap)"""
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

        # 3.1 Meta-SHAP Beeswarm (Summary of feature impact)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Determine strict top 20 features to avoid clutter if many features
        mean_abs_shap = beeswarm_df.groupby('Feature')['SHAP_Value'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
        top_features = mean_abs_shap.index[:20] 
        
        # Filter beeswarm data for top features only, to fix order
        beeswarm_plot_data = beeswarm_df[beeswarm_df['Feature'].isin(top_features)]
        
        sns.stripplot(data=beeswarm_plot_data, x='SHAP_Value', y='Feature', hue='Value', 
                      palette='coolwarm', alpha=0.4, size=3, order=top_features, ax=ax1, jitter=True)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Figure 3(a): Meta-SHAP Global Summary (Uncertainty Drivers)', fontsize=14, fontweight='bold')
        ax1.legend(title='Feature Value', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax1.grid(True, axis='x', linestyle='--', alpha=0.5)

        # 3.2 Site-wise Importance Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        # Filter numerical site columns
        site_cols = [c for c in site_wise_df.columns if 'Site' in c]
        if not site_cols: # Fallback if no site columns found
            ax2.text(0.5, 0.5, "No Site-wise Data Available", ha='center')
        else:
            df_plot = site_wise_df.set_index('Feature')[site_cols]
            # Use same top features as beeswarm
            valid_top_features = [f for f in top_features if f in df_plot.index]
            sns.heatmap(df_plot.loc[valid_top_features], cmap='YlGnBu', annot=True, fmt='.3f', ax=ax2)
            ax2.set_title('Figure 3(b): Site-specific Variable Sensitivity', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/Fig3_Global_Explainability.png', dpi=300)
        plt.close()

    def plot_fig4_temporal_case_study(self, temporal_df, case_shap_series):
        """Figure 4: Temporal & Local Insight (Lag Heatmap & Outlier Waterfall)"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # 4.1 Temporal Lag Importance Map
        ax1 = axes[0]
        # Transpose so features are on Y-axis, Lags on X-axis
        if temporal_df is not None:
            sns.heatmap(temporal_df.T, cmap='rocket_r', ax=ax1)
            ax1.set_title('Figure 4(a): Temporal Lag Importance (Feature x Time)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time Lag (Days before prediction)')
            ax1.set_ylabel('Input Variables')
        else:
             ax1.text(0.5, 0.5, "No Temporal Data", ha='center')

        # 4.2 Local Waterfall Plot for a High Uncertainty Case
        ax2 = axes[1]
        if case_shap_series is not None:
            # case_shap_series: pd.Series index=Feature, values=SHAP
            # Sort by absolute value
            sorted_case = case_shap_series.reindex(case_shap_series.abs().sort_values().index)
            # Take top 15 features
            sorted_case = sorted_case.tail(15)
            
            colors = ['red' if x > 0 else 'blue' for x in sorted_case.values]
            ax2.barh(sorted_case.index, sorted_case.values, color=colors, alpha=0.7)
            ax2.set_title('Figure 4(b): Local Explanation for High-Risk Event', fontsize=14, fontweight='bold')
            ax2.set_xlabel('SHAP Value (Contribution to Uncertainty)')
            ax2.axvline(x=0, color='black', linewidth=0.8)
            ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
        else:
            ax2.text(0.5, 0.5, "No Case Data", ha='center')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/Fig4_Temporal_Local_Insight.png', dpi=300)
        plt.close()
