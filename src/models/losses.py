
import torch
import numpy as np
from typing import Tuple, Dict

def gaussian_nll_loss(y_true: torch.Tensor, y_pred_mean: torch.Tensor, 
                      y_pred_var: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood Loss for Heteroscedastic Uncertainty
    
    Phase 2: Model outputs Mean and Variance directly.
    Captures heteroscedasticity where uncertainty increases with higher Chl-a.
    
    Loss = 0.5 * log(2*pi*sigma^2) + 0.5 * (y - mu)^2 / sigma^2
    
    Args:
        y_true: (batch_size, 1) - 실제 Chl-a 값
        y_pred_mean: (batch_size, 1) - 예측 평균
        y_pred_var: (batch_size, 1) - 예측 분산 (항상 양수)
        eps: Numerical stability constant
    
    Returns:
        loss: Scalar tensor
    """
    # Ensure variance is positive and stable
    y_pred_var = torch.clamp(y_pred_var, min=eps)
    
    # Gaussian NLL: -log P(y|μ, σ²)
    nll = 0.5 * torch.log(2 * np.pi * y_pred_var) + \
          0.5 * (y_true - y_pred_mean) ** 2 / y_pred_var
    
    return torch.mean(nll)

def adaptive_decorrelation_nll_loss(y_true: torch.Tensor, y_pred_mean: torch.Tensor, 
                                    y_pred_var: torch.Tensor, 
                                    alpha: float = 0.3, lambd: float = 0.1, 
                                    eps: float = 1e-6) -> torch.Tensor:
    """
    Adaptive-Detach Loss: Enforces independence between Mean and Variance.
    
    Loss = GaussianNLL + lambda * max(0, |Corr(detach(mu), sigma^2)| - alpha)
    
    1. Detach mean to prevent gradient flow from penalty to mean tower.
    2. If correlation is less than alpha (margin), penalty is 0.
    """
    # 1. Base Gaussian NLL
    nll_loss = gaussian_nll_loss(y_true, y_pred_mean, y_pred_var, eps)
    
    if lambd == 0:
        return nll_loss

    # 2. Adaptive Decorrelation Penalty
    # detach() mean to prevent gradient flow from penalty to mean tower
    mean_detached = y_pred_mean.detach()
    var_pred = y_pred_var
    
    # Calculate Correlation
    # Batch 단위 상관계수 계산
    if mean_detached.size(0) > 1:
        mean_centered = mean_detached - mean_detached.mean()
        var_centered = var_pred - var_pred.mean()
        
        covariance = (mean_centered * var_centered).mean()
        std_mean = torch.sqrt((mean_centered ** 2).mean() + eps)
        std_var = torch.sqrt((var_centered ** 2).mean() + eps)
        
        correlation = covariance / (std_mean * std_var + eps)
        
        # Penalty: max(0, |corr| - alpha)
        corr_penalty = torch.clamp(torch.abs(correlation) - alpha, min=0.0)
    else:
        corr_penalty = torch.tensor(0.0, device=mean_detached.device)
        
    
    total_loss = nll_loss + lambd * corr_penalty
    return total_loss

def iso_nll_loss(y_true: torch.Tensor, y_pred_mean: torch.Tensor, 
                 y_pred_var: torch.Tensor, 
                 lambd_indep: float = 0.1, 
                 lambd_sharp: float = 0.01,
                 alpha: float = 0.3, 
                 eps: float = 1e-6) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Independent Sharpness-Optimized NLL (ISO-NLL)
    
    목표: 
    1. 정확한 확률 분포 학습 (NLL)
    2. 예측값 크기에 따른 구간 확장 억제 (Independence)
    3. 구간의 평균 폭 최소화 (Sharpness)

    Research Contribution:
    - 실무적 차별성: 피크 지점에서 불필요하게 넓은 구간 방지
    - 지표 최적화: CWC (PICP & MPIW) 최적화와 유사 효과
    - 도메인 특화: 데이터 변동성에 기반한 본질적 불확실성 포착
    """
    # 1. Base Gaussian NLL (Reliability의 기초)
    y_pred_var = torch.clamp(y_pred_var, min=eps)
    nll_loss = 0.5 * torch.log(2 * np.pi * y_pred_var) + \
               0.5 * (y_true - y_pred_mean) ** 2 / y_pred_var
    nll_loss = torch.mean(nll_loss)

    # 2. Independence Penalty (Detach-based Correlation)
    # 분산이 단순히 평균의 크기를 따라가지 않도록 분리
    if lambd_indep > 0:
        mean_detached = y_pred_mean.detach()
        if mean_detached.size(0) > 1:
            mean_centered = mean_detached - mean_detached.mean()
            var_centered = y_pred_var - y_pred_var.mean()
            
            covariance = (mean_centered * var_centered).mean()
            std_mean = torch.sqrt((mean_centered ** 2).mean() + eps)
            std_var = torch.sqrt((var_centered ** 2).mean() + eps)
            
            correlation = covariance / (std_mean * std_var + eps)
            p_corr = torch.clamp(torch.abs(correlation) - alpha, min=0.0)
        else:
            p_corr = torch.tensor(0.0, device=y_pred_mean.device)
    else:
        p_corr = torch.tensor(0.0, device=y_pred_mean.device)

    # 3. Sharpness Penalty (Width Minimization)
    # 구간의 폭(분산)이 작을수록 보상 (로그 스케일로 폭주 방지)
    # Note: Using log(var) puts heavy penalty on very large variances, 
    # but allows small variances (negative log). 
    if lambd_sharp > 0:
        # Stabilization: log(var + eps) prevents -inf
        p_width = torch.mean(torch.log(y_pred_var + 1e-8))
    else:
        p_width = torch.tensor(0.0, device=y_pred_mean.device)
        
    # 최종 손실 함수
    total_loss = nll_loss + (lambd_indep * p_corr) + (lambd_sharp * p_width)
    
    # Logs for monitoring
    logs = {
        "loss_nll": nll_loss.item(),
        "loss_p_corr": p_corr.item(),
        "loss_p_width": p_width.item(),
        "total_loss": total_loss.item()
    }
    
    return total_loss, logs


def beta_nll_loss(y_true: torch.Tensor, y_pred_mean: torch.Tensor,
                  y_pred_var: torch.Tensor, beta: float = 0.5,
                  eps: float = 1e-6) -> torch.Tensor:
    """
    beta-NLL (Seitzer et al., ICLR 2022) — ablation baseline.

    Reweights each sample's Gaussian NLL by a stop-gradient power of the
    predicted variance, sigma^{2*beta}, so that poorly fit (high-variance)
    regions are not starved of mean-fitting gradient.
      beta = 0.0 -> ~MSE behavior, beta = 1.0 -> standard NLL.
    Default beta = 0.5 per the original paper.
    """
    y_pred_var = torch.clamp(y_pred_var, min=eps)
    nll = 0.5 * torch.log(2 * np.pi * y_pred_var) + \
          0.5 * (y_true - y_pred_mean) ** 2 / y_pred_var
    weight = (y_pred_var.detach()) ** beta  # stop-gradient weighting
    return torch.mean(weight * nll)


def faithful_hr_loss(y_true: torch.Tensor, y_pred_mean: torch.Tensor,
                     y_pred_var: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Faithful Heteroscedastic Regression (Stirn et al., AISTATS 2023) — ablation
    baseline, adapted to the Decoupled Architecture.

    Mean path is trained with homoscedastic squared error (variance cannot
    rescale the mean gradient); variance path is a Gaussian NLL on a detached
    mean (variance learning cannot perturb the mean). This realizes the
    stop-gradient principle of faithful regression on the separate towers.
    """
    y_pred_var = torch.clamp(y_pred_var, min=eps)
    mean_loss = torch.mean((y_true - y_pred_mean) ** 2)
    mu_det = y_pred_mean.detach()
    var_nll = 0.5 * torch.log(2 * np.pi * y_pred_var) + \
              0.5 * (y_true - mu_det) ** 2 / y_pred_var
    return mean_loss + torch.mean(var_nll)


def iso_der_loss(y_true: torch.Tensor,
                 gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                 lambd_indep: float = 0.1, lambd_sharp: float = 0.05,
                 margin_alpha: float = 0.3, coeff: float = 0.01) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Revised ISO-DER Loss
    
    Stabilization Fixes:
    1. Alpha > 1.1 safeguard (via Softplus + 1.1).
    2. Aleatoric Variance = Beta / (Alpha - 1).
    3. Independence: Corr(Gamma, Log(AleatoricVar)).
    4. Sharpness: Minimize Mean(AleatoricVar).
    """
    
    # 1. Parameter Safeguards
    # Ensure alpha > 1.1 to avoid division by zero or negative variance
    # Assuming model outputs Softplus(alpha), we add offset here OR in model.
    # To be safe, we reinforce it here.
    alpha_safe = torch.nn.functional.softplus(alpha) + 1.1
    nu_safe = torch.nn.functional.softplus(nu) + 1e-6
    beta_safe = torch.nn.functional.softplus(beta) + 1e-6
    
    # 2. DER Loss (NLL + KL)
    # NLL
    two_beta_lambda = 2.0 * beta_safe * (1.0 + nu_safe)
    
    # log(pi/nu) term
    term1 = 0.5 * torch.log(torch.tensor(np.pi, device=gamma.device) / nu_safe)
    # - alpha * log(2*beta)
    term2 = - alpha_safe * torch.log(2.0 * beta_safe)
    # + (alpha + 0.5) * log(nu * (y - gamma)^2 + 2*beta)
    term3 = (alpha_safe + 0.5) * torch.log(nu_safe * (y_true - gamma)**2 + 2.0 * beta_safe)
    # + log_gamma(alpha) - log_gamma(alpha + 0.5)
    term4 = torch.lgamma(alpha_safe) - torch.lgamma(alpha_safe + 0.5)
    
    nll = term1 + term2 + term3 + term4
    nll_loss = torch.mean(nll)

    # KL Regularizer
    # Penalize divergence from Prior (Gamma=y, Nu=?, Alpha=?, Beta=?) 
    # Standard DER Regularizer: Error * KL
    # KL approx: 2*nu + alpha (Minimizing evidence on errors)
    # Use simple evidence regulizer as proxy if full KL is unstable
    error = torch.abs(y_true - gamma)
    kl_proxy = 2.0 * nu_safe + alpha_safe 
    reg_loss = torch.mean(error * kl_proxy)
    
    der_base_loss = nll_loss + coeff * reg_loss

    # 3. ISO Penalties
    # Calculate Moments
    # Aleatoric Variance: Beta / (Alpha - 1)
    aleatoric_var = beta_safe / (alpha_safe - 1.0)
    
    # Log Variance for stability (and better correlation scale)
    log_aleatoric = torch.log(aleatoric_var + 1e-8)

    # A. Independence Penalty (P_corr)
    # Correlation between Mean (Gamma) and Log(Aleatoric Variance)
    if lambd_indep > 0:
        mean_detached = gamma.detach()
        if mean_detached.size(0) > 1:
            mean_centered = mean_detached - mean_detached.mean()
            # Use Log Variance for correlation to handle scale differences
            var_centered = log_aleatoric - log_aleatoric.mean()
            
            covariance = (mean_centered * var_centered).mean()
            std_mean = torch.sqrt((mean_centered ** 2).mean() + 1e-8)
            std_var = torch.sqrt((var_centered ** 2).mean() + 1e-8)
            
            correlation = covariance / (std_mean * std_var + 1e-8)
            p_corr = torch.clamp(torch.abs(correlation) - margin_alpha, min=0.0)
        else:
            p_corr = torch.tensor(0.0, device=gamma.device)
    else:
        p_corr = torch.tensor(0.0, device=gamma.device)

    # B. Sharpness Penalty (P_width)
    # Minimize expected Aleatoric Variance (or log variance)
    # Direct minimization of variance prevents explosion.
    if lambd_sharp > 0:
        # Penalize large variances. Using Log scale is gentler but direct Mean is stronger.
        # Let's use Mean of Log Variance to be consistent with NLL scale 
        # OR Mean of Variance. 
        # Experiment 3 issue was "Explosion", so penalizing Magnitude is key.
        # But NLL already penalizes small variance if error is large.
        # Sharpness penalizes large variance regardless of error.
        # Use simple mean(aleatoric_var) might be unstable if outliers exist.
        # Use mean(log(aleatoric_var))
        p_width = torch.mean(log_aleatoric)
    else:
        p_width = torch.tensor(0.0, device=gamma.device)

    total_loss = der_base_loss + (lambd_indep * p_corr) + (lambd_sharp * p_width)

    logs = {
        "loss_der_nll": nll_loss.item(),
        "loss_der_kl": reg_loss.item(),
        "loss_iso_corr": p_corr.item(),
        "loss_iso_width": p_width.item(),
        "mean_alpha": alpha_safe.mean().item(),
        "mean_beta": beta_safe.mean().item(), # Monitor Beta
        "mean_aleatoric": aleatoric_var.mean().item(), # Monitor Variance
        "total_loss": total_loss.item()
    }

    return total_loss, logs

def _kl_divergence_nig(gamma, nu, alpha, beta):
    # KL divergence between NIG(gamma, nu, alpha, beta) and NIG(0, 0, 1, 0) ??
    # Usually KL to a Prior. Prior often NIG(gamma=0, nu=eps, alpha=1, beta=eps?) or similar.
    # Using standard DER implementation reference.
    # Assuming Prior: Gamma=0, Nu=1, Alpha=1, Beta=1 (Uniform-ish?)
    
    # Using simple approximation or just the Data-Evidence reg.
    # For now, implementing standard DER KL:
    # KL[NIG(g,v,a,b) || NIG(0,0,1,0)] is not well defined?
    # Reference Amini et al code:
    # KL = ...
    
    # Let's use simplified regularizer: Evidence = 2*nu + alpha
    # Use the standard implementation from existing repos if possible.
    # Or strict KL formula.
    
    ones = torch.ones_like(alpha)
    gamma_p = torch.zeros_like(gamma)
    nu_p = torch.ones_like(nu) # Prior nu=1? Or small? Amini uses nu
    alpha_p = torch.ones_like(alpha)
    beta_p = torch.ones_like(beta) # Prior beta?
    
    # Actually, simpler to just return KL components
    # KL = 0.5 * (gamma-gamma_p)^2 * nu
    # But let's stick to a robust implementation.
    
    # Minimal KL Reg: 
    # Penalize large evidence on errors.
    # Evidence = 2 * nu + alpha
    # This is handled by coeff * error * KL logic outside.
    # We just need the term.
    
    # Standard DER KL (Approx):
    kl = 2.0 * nu + alpha 
    return kl 


