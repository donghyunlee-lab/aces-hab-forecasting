
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple

class EvaluationMetrics:
    """Compute time series forecasting evaluation metrics"""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        epsilon = 1e-8
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared (결정계수)"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    @staticmethod
    def rmse_sd_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """RMSE / Standard Deviation"""
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        sd = np.std(y_true)
        return rmse / sd if sd != 0 else float('inf')

    @staticmethod
    def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Nash-Sutcliffe Efficiency
        NSE = 1 - (Sum of squared errors) / (Sum of squared deviations)
        """
        mean_true = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - mean_true) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    @staticmethod
    def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Kling-Gupta Efficiency
        KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
        """
        if len(y_true) < 2: return 0.0
        r = np.corrcoef(y_true, y_pred)[0, 1]
        std_true, std_pred = np.std(y_true), np.std(y_pred)
        mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
        
        alpha = std_pred / std_true if std_true != 0 else 0.0
        beta = mean_pred / mean_true if mean_true != 0 else 0.0
        
        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    @staticmethod
    def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        """Prediction Interval Coverage Probability"""
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return coverage

    @staticmethod
    def pinaw(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray = None) -> float:
        """Prediction Interval Normalized Average Width"""
        width = upper - lower
        normalization = np.std(y_true) if y_true is not None else 1.0
        return np.mean(width) / normalization if normalization != 0 else 0.0

    @staticmethod
    def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
        """Mean Prediction Interval Width"""
        return np.mean(upper - lower)

    @staticmethod
    def cwc(picp: float, mpiw: float, confidence_level: float = 0.95, eta: float = 50.0) -> float:
        """
        Coverage Width-based Criterion
        CWC = MPIW * (1 + gamma * exp(eta * (confidence_level - PICP)))
        """
        gamma = 1 if picp < confidence_level else 0
        return mpiw * (1 + gamma * np.exp(eta * (confidence_level - picp)))

    @staticmethod
    def ks_test(y_true: np.ndarray, pred_mean: np.ndarray, pred_var: np.ndarray) -> float:
        """Kolmogorov-Smirnov test p-value"""
        residuals = y_true - pred_mean
        standardized = residuals / np.sqrt(pred_var + 1e-6)
        ks_stat, p_value = stats.kstest(standardized, 'norm')
        return p_value

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Percent Bias"""
        numerator = np.sum(y_true - y_pred)
        denominator = np.sum(y_true)
        return 100.0 * (numerator / denominator) if denominator != 0 else 0.0

    @staticmethod
    def corr_pred_width(y_pred: np.ndarray, interval_width: np.ndarray) -> float:
        """Correlation between Predicted Mean and CI Width"""
        if len(y_pred) < 2: return 0.0
        corr = np.corrcoef(y_pred, interval_width)[0, 1]
        return corr if not np.isnan(corr) else 0.0

def calculate_comprehensive_metrics(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate comprehensive metrics for global and site-specific performance
    
    Args:
        results_df: DataFrame with columns ['Actual', 'Predicted_Mean', 'CI_Lower', 'CI_Upper', 
                                            'Interval_Width', 'Site' (optional)]
    
    Returns:
        global_metrics_df: DataFrame with columns [Metric, Value, Unit]
        site_metrics_df: DataFrame with columns [Site, R2, RMSE, NSE, PICP, PINAW, MPIW, Corr_Pred_Width]
    """
    # Extract data
    y_true = results_df['Actual'].values
    y_pred = results_df['Predicted_Mean'].values
    lower = results_df['CI_Lower'].values
    upper = results_df['CI_Upper'].values
    interval_width = results_df['Interval_Width'].values
    
    # ====== GLOBAL METRICS ======
    global_metrics = []
    
    # General Performance Metrics
    r2 = EvaluationMetrics.r2_score(y_true, y_pred)
    rmse = EvaluationMetrics.rmse(y_true, y_pred)
    nse = EvaluationMetrics.nse(y_true, y_pred)
    rmse_sd = EvaluationMetrics.rmse_sd_ratio(y_true, y_pred)
    mae = EvaluationMetrics.mae(y_true, y_pred)
    pbias = EvaluationMetrics.pbias(y_true, y_pred)
    kge = EvaluationMetrics.kge(y_true, y_pred)
    mape = EvaluationMetrics.mape(y_true, y_pred)
    
    # Uncertainty Metrics
    picp = EvaluationMetrics.picp(y_true, lower, upper)
    pinaw = EvaluationMetrics.pinaw(lower, upper, y_true)
    mpiw = EvaluationMetrics.mpiw(lower, upper)
    corr_pred_width = EvaluationMetrics.corr_pred_width(y_pred, interval_width)
    cwc = EvaluationMetrics.cwc(picp, mpiw, confidence_level=0.95)
    
    # Build global metrics DataFrame
    global_metrics_data = [
        {'Metric': 'R²', 'Value': r2, 'Unit': 'dimensionless'},
        {'Metric': 'RMSE', 'Value': rmse, 'Unit': 'mg/m³'},
        {'Metric': 'NSE', 'Value': nse, 'Unit': 'dimensionless'},
        {'Metric': 'KGE', 'Value': kge, 'Unit': 'dimensionless'},
        {'Metric': 'RMSE/SD', 'Value': rmse_sd, 'Unit': 'dimensionless'},
        {'Metric': 'MAE', 'Value': mae, 'Unit': 'mg/m³'},
        {'Metric': 'MAPE', 'Value': mape, 'Unit': '%'},
        {'Metric': 'PBIAS', 'Value': pbias, 'Unit': '%'},
        {'Metric': 'PICP', 'Value': picp, 'Unit': 'dimensionless'},
        {'Metric': 'PINAW', 'Value': pinaw, 'Unit': 'dimensionless'},
        {'Metric': 'MPIW', 'Value': mpiw, 'Unit': 'mg/m³'},
        {'Metric': 'CWC', 'Value': cwc, 'Unit': 'dimensionless'},
        {'Metric': 'Corr(Pred, Width)', 'Value': corr_pred_width, 'Unit': 'dimensionless'},
    ]
    
    
    # ====== PEAK ANALYSIS (Top 10%) ======
    # Assess performance on extreme events (Algal Blooms)
    peak_threshold = np.percentile(y_true, 90)
    peak_mask = y_true >= peak_threshold
    
    if np.sum(peak_mask) > 0:
        y_peak = y_true[peak_mask]
        pred_peak = y_pred[peak_mask]
        lower_peak = lower[peak_mask]
        upper_peak = upper[peak_mask]
        
        peak_rmse = EvaluationMetrics.rmse(y_peak, pred_peak)
        peak_picp = EvaluationMetrics.picp(y_peak, lower_peak, upper_peak)
        peak_mpiw = EvaluationMetrics.mpiw(lower_peak, upper_peak)
        peak_cwc = EvaluationMetrics.cwc(peak_picp, peak_mpiw)
        
        global_metrics_data.extend([
            {'Metric': 'Peak RMSE', 'Value': peak_rmse, 'Unit': 'mg/m³'},
            {'Metric': 'Peak PICP', 'Value': peak_picp, 'Unit': 'dimensionless'},
            {'Metric': 'Peak MPIW', 'Value': peak_mpiw, 'Unit': 'mg/m³'},
            {'Metric': 'Peak CWC', 'Value': peak_cwc, 'Unit': 'dimensionless'}
        ])
    
    global_metrics_df = pd.DataFrame(global_metrics_data)
    
    # ====== SITE-SPECIFIC METRICS ======
    site_metrics_data = []
    
    if 'Site' in results_df.columns:
        sites = results_df['Site'].unique()
        # site_id_to_name is unused but conceptually exists
        
        for site in sites:
            site_df = results_df[results_df['Site'] == site]
            site_y_true = site_df['Actual'].values
            site_y_pred = site_df['Predicted_Mean'].values
            site_lower = site_df['CI_Lower'].values
            site_upper = site_df['CI_Upper'].values
            site_interval_width = site_df['Interval_Width'].values
            
            # Calculate metrics for this site
            site_r2 = EvaluationMetrics.r2_score(site_y_true, site_y_pred)
            site_rmse = EvaluationMetrics.rmse(site_y_true, site_y_pred)
            site_nse = EvaluationMetrics.nse(site_y_true, site_y_pred)
            site_picp = EvaluationMetrics.picp(site_y_true, site_lower, site_upper)
            site_pinaw = EvaluationMetrics.pinaw(site_lower, site_upper, site_y_true)
            site_mpiw = EvaluationMetrics.mpiw(site_lower, site_upper)
            site_corr = EvaluationMetrics.corr_pred_width(site_y_pred, site_interval_width)
            
            site_metrics_data.append({
                'Site': site,
                'R2': site_r2,
                'RMSE': site_rmse,
                'NSE': site_nse,
                'PICP': site_picp,
                'PINAW': site_pinaw,
                'MPIW': site_mpiw,
                'Corr_Pred_Width': site_corr
            })
    else:
        # If no Site column, create a single row with 'Global'
        site_metrics_data.append({
            'Site': 'Global',
            'R2': r2,
            'RMSE': rmse,
            'NSE': nse,
            'PICP': picp,
            'PINAW': pinaw,
            'MPIW': mpiw,
            'Corr_Pred_Width': corr_pred_width
        })
    
    site_metrics_df = pd.DataFrame(site_metrics_data)
    
    return global_metrics_df, site_metrics_df
