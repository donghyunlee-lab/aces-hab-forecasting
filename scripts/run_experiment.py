
import os
import argparse
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessor import DataPreprocessor
from src.models.builder import get_model
from src.models.inference import predict_uncertainty
from src.training.trainer import HABTrainer
from src.evaluation.metrics import EvaluationMetrics, calculate_comprehensive_metrics
from src.evaluation.shap import MetaSHAP
from src.evaluation.uncertainty import LaplaceApproximation, AdaptiveCP
from src.visualization.plots import HABPublicationVisualizer, plot_beeswarm_custom
from src.utils.constants import translate_feature_names, get_device, cleanup_memory
from src.utils.reproducibility import set_seed, log_environment

def create_config(model_type: str = 'iTransformer', uq_method: str = 'Laplace', 
                  train_start_year: int = 2021, train_end_year: int = 2022,
                  site_names: List[str] = None) -> dict:
    """Create comprehensive configuration with dynamic paths"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create dynamic subdirectories
    results_subdir = f"{model_type}_{uq_method}"
    metrics_dir = f'{base_path}/results/metrics/{results_subdir}'
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Default sites if None
    default_sites = ['공주', '대청호', '갑천', '부여', '용담호']
    selected_sites = site_names if site_names is not None else default_sites

    config = {
        # Experiment Settings
        'model_type': model_type,
        'uq_method': uq_method,

        # Data - Phase 1: Panel Data Mode
        'data_path': f'{base_path}/data/imputed_daily_data.csv',
        'site_names': selected_sites,  # 5 sites (Configurable)
        'use_panel_data': True,
        
        # Period Selection (Configurable)
        'train_period': (f'{train_start_year}-01-01', f'{train_end_year}-12-31'),
        'val_period':   ('2023-01-01', '2023-12-31'), # 1 year
        'test_period':  ('2024-01-01', '2024-12-31'), # 1 year

        'apply_log_transform': True,
        'target_col': '클로로필-a (mg/㎥)',

        # Model architecture
        'seq_len': 30,
        'hidden_dim': 64,
        'n_heads': 4,
        'n_layers': 2,
        'dropout_rate': 0.3,
        'output_dim': None,

        # Training
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'mc_samples': 1,  # Phase 2: Use Gaussian NLL
        'l2_penalty': 1e-4,
        'patience': 15,

        # Forecast
        'forecast_days': 365,

        # Paths (Dynamic)
        'model_save_path': f'{base_path}/models/best_{results_subdir}_model.pth',
        'results_save_path': f'{metrics_dir}/results_summary.csv',
        'metrics_save_path': f'{metrics_dir}/metrics_summary.json',
        'plot_save_path': f'{base_path}/results/visualizations/{results_subdir}_predictions.png',
        
        # Comprehensive Metrics Paths
        'global_metrics_path': f'{metrics_dir}/global_performance_metrics.csv',
        'site_metrics_path': f'{metrics_dir}/site_specific_metrics.csv',
        
        'shap_analysis': False,  # Default: off, enable with --run-shap flag

        # Ablation: beta-NLL exponent (used only when uq_method == 'BetaNLL')
        'beta_nll': 0.5,

        # Reproducibility: fixed RNG seed for this run (ensemble members override)
        'seed': 42
    }
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['plot_save_path']), exist_ok=True)

    return config

def run_experiment(config: Dict):
    """Main training and evaluation pipeline"""
    # Seed all RNGs first so model init, shuffling, and ensembling are reproducible.
    seed = set_seed(config.get('seed', 42))
    print(f"[REPRODUCIBILITY] seed={seed} | env={log_environment()}")
    device = get_device()
    print(f"Using device: {device}")

    # ====== STEP 1: DATA PREPARATION ======
    print("\n[STEP 1] Phase 1: Panel Data Preparation...")
    preprocessor = DataPreprocessor(
        data_path=config['data_path'],
        site_names=config['site_names'],
        apply_log_transform=config.get('apply_log_transform', True),
        use_panel_data=config.get('use_panel_data', True)
    )

    df = preprocessor.load_data()
    df_normalized, feature_names = preprocessor.preprocess(df)
    data_split = preprocessor.split_train_test(
        df, df_normalized, 
        seq_len=config['seq_len'],
        train_period=config['train_period'],
        val_period=config['val_period'],
        test_period=config['test_period']
    )

    # Convert to tensors
    X_train = torch.from_numpy(data_split['X_train']).float().to(device)
    y_train = torch.from_numpy(data_split['y_train']).float().to(device)
    X_val = torch.from_numpy(data_split['X_val']).float().to(device)
    y_val = torch.from_numpy(data_split['y_val']).float().to(device)
    X_test = torch.from_numpy(data_split['X_test']).float().to(device)
    y_test = torch.from_numpy(data_split['y_test']).float().to(device)
    
    train_station_ids = None
    val_station_ids = None
    test_station_ids = None
    
    if config['use_panel_data']:
        train_station_ids = torch.from_numpy(data_split['train_station_ids']).long().to(device)
        val_station_ids = torch.from_numpy(data_split['val_station_ids']).long().to(device)
        test_station_ids = torch.from_numpy(data_split['test_station_ids']).long().to(device)

    # ====== STEP 2: BUILD MODEL ======
    print(f"\n[STEP 2] Building Model: {config.get('model_type', 'iTransformer')}...")
    n_stations = len(config['site_names']) if config['use_panel_data'] else 1
    model = get_model(config, n_features=len(feature_names), n_stations=n_stations).to(device)

    # ====== STEP 3: TRAINING ======
    uq_method = config.get('uq_method', 'Laplace')
    
    if uq_method == 'Ensemble' and config.get('ensemble_paths'):
        print(f"\n[STEP 3] Loading Deep Ensemble ({len(config['ensemble_paths'])} members)...")
        models = []
        for path in config['ensemble_paths']:
            # Create fresh model instance
            m = get_model(config, n_features=len(feature_names), n_stations=n_stations).to(device)
            if os.path.exists(path):
                m.load_state_dict(torch.load(path, map_location=device))
                models.append(m)
            else:
                print(f"Warning: Member path {path} not found. Skipping.")
        
        if not models:
            raise ValueError("No ensemble members loaded!")
        
        # Override 'model' variable with list of models for inference
        model = models 
        
    elif config.get('skip_training', False) and os.path.exists(config['model_save_path']):
        print(f"\n[STEP 3] Skipping Training (Loading from {config['model_save_path']})...")
        model.load_state_dict(torch.load(config['model_save_path'], map_location=device))
    else:
        trainer = HABTrainer(model, config)
        model = trainer.train(X_train, y_train, X_val, y_val, train_station_ids, val_station_ids)

    # ====== STEP 4: LAPLACE APPROXIMATION ======
    # Only if NOT Ensemble and needed
    if uq_method in ['Laplace', 'ACP'] and not isinstance(model, list):
        print("\n[STEP 4] Computing Laplace Approximation...")
        model.eval()
        laplace = LaplaceApproximation(model, prior_var=1.0)
        
        train_station_ids_laplace = train_station_ids[:min(100, len(train_station_ids))] if train_station_ids is not None else None
        
        laplace.compute_diagonal_hessian(
            X_train[:min(100, len(X_train))], 
            y_train[:min(100, len(y_train))],
            n_samples=min(50, len(X_train)),
            station_ids=train_station_ids_laplace
        )
    else:
        laplace = None # Ensemble or Point or MCDO doesn't use this Laplace object (MCDO has dropout)
    
    posterior_var = laplace.get_posterior_variance() if laplace is not None else 0.0
    if uq_method != 'Ensemble': # Only print for single model
        print(f"  Posterior variance (Laplace): {posterior_var:.6f}")

    # ====== STEP 5: EVALUATION ======
    print("\n[STEP 5] Phase 2: Evaluation on Test Set...")
    uq_method = config.get('uq_method', 'Laplace')
    
    pred_mean, pred_var = predict_uncertainty(
        model, X_test, 
        station_ids=test_station_ids, 
        method=uq_method if uq_method != 'ACP' else 'Laplace',  # ACP uses Laplace as base
        laplace=laplace if uq_method in ['Laplace', 'ACP'] else None,
        n_samples=50
    )
    
    # Process predictions (Denormalize)
    pred_mean_np = pred_mean.cpu().numpy().squeeze(1)
    # Variance denormalization logic (approximate)
    target_idx = preprocessor.target_idx
    data_min = preprocessor.scaler.data_min_[target_idx]
    data_max = preprocessor.scaler.data_max_[target_idx]
    data_range = data_max - data_min
    
    var_denorm = pred_var.cpu().numpy().squeeze(1) * (data_range ** 2)
    
    # Needs full matrix for inverse transform
    dummy_full = np.zeros((len(pred_mean_np), len(feature_names)))
    dummy_full[:, target_idx] = pred_mean_np
    pred_mean_denorm = preprocessor.inverse_transform(dummy_full)[:, target_idx]
    
    if preprocessor.apply_log_transform:
        var_denorm = var_denorm * ((pred_mean_denorm + 1) ** 2) # Delta method approx
    
    sigma = np.sqrt(var_denorm)
    
    # Apply ACP calibration if requested
    if uq_method == 'ACP' or config.get('apply_acp', False):
        print("  Applying Adaptive Conformal Prediction (ACP)...")
        alpha_target = config.get('acp_alpha', 0.05)  
        print(f"  ACP Target Alpha: {alpha_target} (Coverage Target: {1-alpha_target:.2%})")
        acp = AdaptiveCP(alpha=alpha_target)
        
        # Get validation predictions for calibration
        # Get validation predictions for calibration
        # ACP SPLIT: Use 20% of Validation Set for Calibration
        len_val = len(X_val)
        calib_size = int(len_val * 0.2)
        
        # We take the LAST 20% for calibration (assuming temporal order matters less here or simply split)
        # Or random? Temporal split is safer.
        # Let's use the first 20% for calibration to mimic "past data adjusting future"
        
        X_calib = X_val[:calib_size]
        y_calib_tensor = y_val[:calib_size]
        
        if val_station_ids is not None:
            station_ids_calib = val_station_ids[:calib_size]
        else:
            station_ids_calib = None
            
        print(f"  ACP Split: Using {calib_size}/{len_val} validation samples for calibration.")
        
        val_pred_mean, val_pred_var = predict_uncertainty(
            model, X_calib, station_ids=station_ids_calib,
            method='Laplace' if isinstance(model, torch.nn.Module) and uq_method != 'Ensemble' else uq_method,
            laplace=laplace
        )
        val_pred_mean_np = val_pred_mean.cpu().numpy().squeeze(1)
        val_var_denorm = val_pred_var.cpu().numpy().squeeze(1) * (data_range ** 2)
        
        dummy_val = np.zeros((len(val_pred_mean_np), len(feature_names)))
        dummy_val[:, target_idx] = val_pred_mean_np
        val_pred_denorm = preprocessor.inverse_transform(dummy_val)[:, target_idx]
        
        if preprocessor.apply_log_transform:
            val_var_denorm = val_var_denorm * ((val_pred_denorm + 1) ** 2)
        
        y_val_np = y_calib_tensor.cpu().numpy().squeeze(1)
        dummy_yval = np.zeros((len(y_val_np), len(feature_names)))
        dummy_yval[:, target_idx] = y_val_np
        y_val_denorm = preprocessor.inverse_transform(dummy_yval)[:, target_idx]
        
        # Calibrate on validation set
        calib_quantile = acp.calibrate(y_val_denorm, val_pred_denorm, val_var_denorm)
        print(f"  Calibrated quantile: {calib_quantile:.4f} (vs 1.96 for standard 95% CI)")
        
        # Apply calibrated intervals to test set
        lower, upper = acp.predict_interval(pred_mean_denorm, var_denorm)
    else:
        lower = pred_mean_denorm - 1.96 * sigma
        upper = pred_mean_denorm + 1.96 * sigma
    
    y_test_np = y_test.cpu().numpy().squeeze(1)
    dummy_y = np.zeros((len(y_test_np), len(feature_names)))
    dummy_y[:, target_idx] = y_test_np
    y_test_denorm = preprocessor.inverse_transform(dummy_y)[:, target_idx]

    # Save Results
    results_df = pd.DataFrame({
        'Date': data_split['test_dates'][:len(pred_mean_denorm)],
        'Actual': y_test_denorm,
        'Predicted_Mean': pred_mean_denorm,
        'Predicted_Std': sigma,
        'CI_Lower': lower,
        'CI_Upper': upper,
        'Interval_Width': upper - lower,
        'Coverage': ((y_test_denorm >= lower) & (y_test_denorm <= upper)).astype(int)
    })
    
    if test_station_ids is not None:
        site_map = {i: name for i, name in enumerate(config['site_names'])}
        # Fix map to only map known indices
        results_df['Station_ID'] = data_split['test_station_ids'][:len(results_df)]
        results_df['Site'] = results_df['Station_ID'].map(site_map)

    results_df.to_csv(config['results_save_path'], index=False)
    print(f"Results saved to {config['results_save_path']}")
    
    # Calculate & Save Metrics
    global_metrics, site_metrics = calculate_comprehensive_metrics(results_df)
    global_metrics.to_csv(config['global_metrics_path'], index=False)
    site_metrics.to_csv(config['site_metrics_path'], index=False)

    # ====== STEP 6: VISUALIZATION ======
    # Use HABPublicationVisualizer
    visualizer = HABPublicationVisualizer(save_dir=os.path.dirname(config['plot_save_path']))
    visualizer.plot_fig1_performance_suite(results_df)
    visualizer.plot_fig2_reliability_suite(results_df)
    
    # ====== STEP 7: SHAP ======
    if config['shap_analysis']:
        print("\n[STEP 7] SHAP Analysis...")
        meta_shap = MetaSHAP(model, feature_names)
        
        # Test station IDs for SHAP
        sid_shap = data_split['test_station_ids']
        
        # Beeswarm
        def target_func(m, v): return m
        df_beeswarm = meta_shap.get_beeswarm_data(data_split['X_test'], target_func, sid_shap)
        
        # Site Heatmap
        df_site = meta_shap.compute_site_wise_importance(data_split['X_test'], target_func, sid_shap)
        visualizer.plot_fig3_global_explainability(df_beeswarm, df_site)
        
        # Temporal
        # Note: Temporal heatmap needs specific implementation in MetaSHAP which calls compute_temporal_importance
        # We'll skip complex temporal plot call for now or use placeholders if not implemented in Visualizer
        # The visualizer has plot_fig4_temporal_case_study.
        
    print("\nExperiment Completed Successfully.")
    # Return results for further analysis
    results_dict = {
        'y_true': y_test_denorm,
        'y_pred': pred_mean_denorm, 
        'y_var': var_denorm,
        'station_ids': test_station_ids
    }
    
    return model, global_metrics, results_df, results_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='iTransformer', 
                       choices=['iTransformer', 'GRU', 'PatchTST', 'TCN'],
                       help='Model type')
    parser.add_argument('--uq', type=str, default='Laplace',
                       help='Uncertainty Quantification method')
    parser.add_argument('--sites', nargs='+', default=None)
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--test-run', action='store_true', help='Run a quick test with 1 epoch')
    parser.add_argument('--run-shap', action='store_true', help='Enable SHAP analysis')
    parser.add_argument('--decoupled', action='store_true', 
                       help='Use decoupled architecture (separate Mean/Var towers)')
    args = parser.parse_args()
    
    config = create_config(model_type=args.model, uq_method=args.uq, site_names=args.sites)
    
    # Update config with CLI args
    config['epochs'] = args.epochs
    config['shap_analysis'] = args.run_shap
    config['decoupled'] = args.decoupled  # Enable decoupled architecture
    
    if args.test_run:
        print("\n[TEST RUN] Overriding configuration for quick testing...")
        config['epochs'] = 1
        config['batch_size'] = 16
        
    run_experiment(config)
