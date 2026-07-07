
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from src.utils.constants import get_device

class LaplaceApproximation:
    """
    Laplace Approximation: 신경망의 사후분포를 가우시안으로 근사

    원리:
    1. MAP(최대사후) 추정값에서 Hessian 계산
    2. 대각 Hessian으로 근사 (계산 효율성)
    3. 사후분포 ~ N(w_map, H^-1)
    
    이를 통해 모델 파라미터의 불확실성을 추정하고,
    예측 불확실성에 통합할 수 있음
    """

    def __init__(self, model: nn.Module, prior_var: float = 1.0):
        self.model = model
        self.prior_var = prior_var
        self.posterior_var = None
        self.hessian_diag = None
        self.n_params = sum(p.numel() for p in model.parameters())
        self.device = get_device()

    def compute_diagonal_hessian(self, X: torch.Tensor, y: torch.Tensor,
                                n_samples: int = None, station_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Diagonal Hessian (Fisher Information Matrix)

        H = E[grad_log_p(y|x,w) * grad_log_p(y|x,w)^T]
        
        Args:
            X: 입력 데이터 (n_samples, seq_len, n_features)
            y: 타겟 데이터 (n_samples, 1) - Phase 2: Chl-a only
            n_samples: 계산에 사용할 샘플 수 (None이면 최대 50개)
            station_ids: (n_samples,) - Phase 1: Station IDs
        
        Returns:
            hessian_diag: 대각 Hessian 벡터
        """
        if n_samples is None:
            n_samples = min(len(X), 50)  # 메모리 효율

        hessian_diag = None
        self.model.train()

        # 배치 단위로 처리
        batch_size = min(10, n_samples)
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            x_batch = X[i:batch_end].to(self.device)
            y_batch = y[i:batch_end].to(self.device)
            
            # Phase 1: Get Station IDs for this batch
            station_batch = None
            if station_ids is not None:
                station_batch = station_ids[i:batch_end].to(self.device)

            self.model.zero_grad()

            # Forward pass - Phase 1 & 2: Pass Station IDs
            pred_mean, pred_var = self.model(x_batch, station_ids=station_batch, mc_samples=1)

            # Phase 2: Gaussian NLL Loss (use function directly, no import needed)
            pred_var_clamped = torch.clamp(pred_var, min=1e-6)
            nll = 0.5 * torch.log(2 * np.pi * pred_var_clamped) + \
                  0.5 * (y_batch - pred_mean) ** 2 / pred_var_clamped
            nll = torch.mean(nll)

            # Compute gradients (score)
            nll.backward(retain_graph=True)

            # Extract diagonal Fisher Information
            batch_hessian = []
            for param in self.model.parameters():
                if param.grad is not None:
                    batch_hessian.append((param.grad ** 2).detach().flatten())

            if batch_hessian:
                batch_hessian_cat = torch.cat(batch_hessian)

                if hessian_diag is None:
                    hessian_diag = batch_hessian_cat.clone()
                else:
                    hessian_diag += batch_hessian_cat

        if hessian_diag is not None:
            hessian_diag = hessian_diag / n_samples
            self.hessian_diag = hessian_diag
            # 사후 분산 = (H + prior_prec * I)^-1 (대각 근사)
            prior_prec = 1.0 / self.prior_var
            self.posterior_var = 1.0 / (hessian_diag + prior_prec + 1e-8)

        self.model.eval()
        return hessian_diag

    def get_posterior_variance(self) -> float:
        """사후분포의 평균 분산 (H^-1)"""
        if self.posterior_var is not None:
            return self.posterior_var.mean().item()
        return 0.1

    def enhance_prediction_uncertainty(self, pred_var: torch.Tensor) -> torch.Tensor:
        """
        Laplace Approximation을 사용하여 예측 불확실성 향상
        
        Args:
            pred_var: 모델의 예측 분산 (batch, n_features)
        
        Returns:
            enhanced_var: Laplace 근사로 향상된 분산
        """
        if self.posterior_var is None:
            return pred_var
        
        # 사후 분산을 스케일링하여 예측 불확실성에 통합
        laplace_uncertainty = self.get_posterior_variance()
        enhancement_factor = 1.0 + laplace_uncertainty * 0.5  # 정밀 튜닝: PICP ~0.95
        
        return pred_var * enhancement_factor


class AdaptiveCP:
    """
    Adaptive Conformal Prediction for calibrated prediction intervals

    ACP adjusts the quantile level dynamically based on the model confidence,
    ensuring valid coverage probability on the test set.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Miscoverage level (e.g., 0.05 for 95% coverage)
        """
        self.alpha = alpha
        self.calibration_scores = None

    def calibrate(self, y_calib: np.ndarray, pred_mean: np.ndarray,
                  pred_var: np.ndarray):
        """
        Calibration using conformal prediction

        Args:
            y_calib: Calibration ground truth
            pred_mean: Predicted mean values
            pred_var: Predicted variance
        """
        # Compute nonconformity scores (absolute residuals normalized by std)
        residuals = np.abs(y_calib - pred_mean)
        pred_std = np.sqrt(pred_var)
        scores = residuals / (pred_std + 1e-6)

        # Find quantile for coverage level
        self.calib_quantile = np.quantile(scores, 1 - self.alpha)

        return self.calib_quantile

    def predict_interval(self, pred_mean: np.ndarray, pred_var: np.ndarray,
                        quantile: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals using calibrated quantile

        Returns:
            lower: Lower bound of interval
            upper: Upper bound of interval
        """
        if quantile is None:
            quantile = self.calib_quantile if self.calib_quantile is not None else 1.96

        pred_std = np.sqrt(pred_var)
        lower = pred_mean - quantile * pred_std
        upper = pred_mean + quantile * pred_std

        return lower, upper


class EnbPI:
    """
    Ensemble Batch Prediction Intervals (EnbPI)
    Uses leave-one-out residuals from ensemble/training to calibrate.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.residuals = []
        self.quantile_val = None

    def fit_residuals(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Store residuals for calibration.
        Ideally uses LOO residuals, but for Phase 3 we use Validation residuals.
        """
        res = np.abs(y_true - y_pred)
        self.residuals.extend(res.tolist())
        
    def calibrate(self):
        """Compute quantile from stored residuals"""
        if not self.residuals:
             return 0.0
        # EnbPI quantile: (1-alpha) quantile of residuals
        self.quantile_val = np.quantile(self.residuals, 1 - self.alpha)
        return self.quantile_val
        
    def predict_interval(self, pred_mean: np.ndarray, quantile: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        EnbPI Interval = Mean +/- Quantile
        (Constant width based on residual distribution)
        """
        if quantile is None:
            quantile = self.quantile_val if self.quantile_val is not None else 0.0
            
        lower = pred_mean - quantile
        upper = pred_mean + quantile
        return lower, upper


class OnlineACP:
    """
    Online Adaptive Conformal Inference (ACI), Gibbs & Candes (2021).

    Unlike the static split-conformal `AdaptiveCP` (one fixed calibration
    quantile), this maintains a time-varying miscoverage level alpha_t and
    updates it after every observation so that long-run coverage tracks the
    target even under distribution shift. Designed for a single temporally
    ordered stream (e.g. one monitoring site over 2024); call once per site.

    Nonconformity score matches AdaptiveCP: s_t = |y_t - mu_t| / (sigma_t + eps).
    Update rule:  alpha_{t+1} = clip(alpha_t + gamma * (alpha_target - err_t), 0, 1)
    where err_t = 1 if the realized point fell outside the interval, else 0.
    """

    def __init__(self, alpha: float = 0.10, gamma: float = 0.02, eps: float = 1e-6):
        """
        Args:
            alpha: target miscoverage (0.10 -> 90% coverage).
            gamma: step size for the alpha_t update (larger = faster adaptation,
                   noisier coverage). Typical 0.005-0.05.
            eps:   std floor for the normalized score.
        """
        self.alpha_target = alpha
        self.gamma = gamma
        self.eps = eps

    def run_stream(self, y, pred_mean, pred_std, warm_scores=None):
        """
        Process one ordered stream. Returns a dict with interval bounds, the
        alpha_t / quantile trajectories, and the realized coverage mask.

        Args:
            y, pred_mean, pred_std: 1-D arrays in temporal order.
            warm_scores: optional 1-D array of calibration scores (e.g. from the
                         validation year) to warm-start the score pool. If None,
                         the pool grows from the stream itself.
        """
        y = np.asarray(y, float)
        mu = np.asarray(pred_mean, float)
        sd = np.asarray(pred_std, float)
        n = len(y)

        pool = [] if warm_scores is None else list(np.asarray(warm_scores, float))
        alpha_t = self.alpha_target

        lower = np.empty(n); upper = np.empty(n)
        covered = np.empty(n, bool)
        alpha_traj = np.empty(n); q_traj = np.empty(n)

        for t in range(n):
            if pool:
                # clip the level to [0,1]: alpha_t<=0 -> max score (widest finite
                # interval), alpha_t>=1 -> min score. Avoids the degenerate
                # infinite interval of vanilla ACI while keeping coverage maximal.
                level = float(np.clip(1.0 - alpha_t, 0.0, 1.0))
                q = float(np.quantile(pool, level))
            else:
                q = 1.96  # cold start before any score is seen

            half = q * (sd[t] + self.eps)
            lower[t] = mu[t] - half
            upper[t] = mu[t] + half

            s_t = abs(y[t] - mu[t]) / (sd[t] + self.eps)
            err_t = 0.0 if s_t <= q else 1.0
            covered[t] = err_t == 0.0
            alpha_traj[t] = alpha_t
            q_traj[t] = q

            alpha_t = float(np.clip(alpha_t + self.gamma * (self.alpha_target - err_t), 0.0, 1.0))
            pool.append(s_t)

        return {
            "lower": lower, "upper": upper, "covered": covered,
            "alpha_t": alpha_traj, "quantile": q_traj,
            "picp": float(covered.mean()),
            "mpiw": float((upper - lower).mean()),
        }

def compute_der_uncertainty(gamma: torch.Tensor, nu: torch.Tensor, 
                            alpha: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Aleatoric and Epistemic Uncertainty from DER parameters
    Distribution: Normal-Inverse-Gamma(gamma, nu, alpha, beta)
    
    Returns:
        aleatoric: E[sigma^2] = beta / (alpha - 1)
        epistemic: Var[mu] = beta / (nu * (alpha - 1))
    """
    # Aleatoric Uncertainty
    aleatoric = beta / (alpha - 1 + 1e-6)
    
    # Epistemic Uncertainty
    epistemic = beta / (nu * (alpha - 1) + 1e-6)
    
    return aleatoric, epistemic
