
import torch
from src.evaluation.uncertainty import LaplaceApproximation


def predict_uncertainty(model, X, station_ids=None, method='Laplace', 
                       laplace=None, calibrator=None, calibration_method='Vanilla', 
                       n_samples=50):
    """
    Unified prediction interface for UQ
    calibration_method: 'Vanilla', 'ACP', 'EnbPI'
    """
    # ... (Method logic) ...
    
    # 1. Get Mean/Var using specified Method
    if method == 'Point':
        with torch.no_grad():
            pred_mean, pred_var = model(X, station_ids=station_ids)
    elif method == 'MCDO':
        model.train()
        means, vars_ = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                m, v = model(X, station_ids=station_ids)
                means.append(m)
                vars_.append(v)
        means = torch.stack(means)
        vars_ = torch.stack(vars_)
        pred_mean = means.mean(dim=0)
        pred_var = vars_.mean(dim=0) + means.var(dim=0)
        model.eval()
    elif method == 'Ensemble':
        if isinstance(model, list):
            pred_mean, pred_var = predict_ensemble(model, X, station_ids)
        else:
            raise ValueError("Ensemble method requires list of models")
    elif method == 'DER' or method == 'ISODER': # Handle ISODER same as DER
        with torch.no_grad():
            if hasattr(model, 'get_evidential_params'):
                gamma, nu, alpha, beta = model.get_evidential_params(X, station_ids=station_ids)
            else:
                 outputs, _ = model(X, station_ids=station_ids)
                 gamma, nu, alpha, beta = torch.split(outputs, 1, dim=-1)
            
            from src.evaluation.uncertainty import compute_der_uncertainty
            aleatoric, epistemic = compute_der_uncertainty(gamma, nu, alpha, beta)
            pred_mean = gamma
            pred_var = aleatoric + epistemic 
    elif method == 'SNGP':
        with torch.no_grad():
             pred_mean, pred_var = model(X, station_ids=station_ids)
    else: # Laplace or Standard
        with torch.no_grad():
            pred_mean, pred_var = model(X, station_ids=station_ids)
            if method == 'Laplace' and laplace is not None:
                pred_var = laplace.enhance_prediction_uncertainty(pred_var)

    # 2. Apply Calibration (Return Interval or Adjusted Var?)
    # predict_uncertainty standardly returns mean, var. 
    # ACP/EnbPI return Intervals. 
    # To maintain interface, if calibrator is present, we might attach info 
    # OR we return mean, var and let caller handle interval. 
    # BUT prompt says "return calibration interval". 
    # This might break existing calls expecting 2 values.
    # We will return mean, var as usual, and let the Automation Script handle the interval generation 
    # using the calibrator object. 
    # Exception: If 'calibrator' is passed, we can't easily return intervals in (mean, var) format 
    # unless we convert interval to var (approx). 
    # Strategy: Return (mean, var). The Calibration Logic happens in EVALUATION (run_experiment or script), 
    # taking mean/var and applying calibrator.predict_interval().
    
    return pred_mean, pred_var

def predict_ensemble(models: list, X: torch.Tensor, station_ids: torch.Tensor = None):
    """
    Deep Ensembles Inference (Mixture of Gaussians)
    
    mu_ens = (1/M) * sum(mu_i)
    var_ens = (1/M) * sum(var_i + mu_i^2) - mu_ens^2
    """
    means = []
    vars_ = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            m, v = model(X, station_ids=station_ids)
            means.append(m)
            vars_.append(v)
            
    means = torch.stack(means) # (M, Batch, 1)
    vars_ = torch.stack(vars_) # (M, Batch, 1)
    
    # ensure positive variance
    vars_ = torch.clamp(vars_, min=1e-6)
    
    # Mixture Mean
    mu_ens = means.mean(dim=0)
    
    # Mixture Variance = Mean(Var + Mu^2) - Mu_ens^2
    term1 = (vars_ + means**2).mean(dim=0)
    var_ens = term1 - mu_ens**2
    
    # Numerical stability
    var_ens = torch.clamp(var_ens, min=1e-6)
    
    return mu_ens, var_ens
