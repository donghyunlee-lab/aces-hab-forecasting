
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class MambaBlock(nn.Module):
    """
    Pure PyTorch Mamba Block
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = (self.d_model + 16) // 16

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        
        self.act = nn.SiLU()

    def ssm(self, x):
        """
        Runs the SSM.
        x: (B, L, D)
        """
        (B, L, D) = x.shape
        
        # Projections
        x_proj = self.x_proj(x) # (B, L, dt_rank + 2*d_state)
        
        delta_rank, B_ssm, C_ssm = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        delta = self.dt_proj(delta_rank) # (B, L, D)
        delta = torch.nn.functional.softplus(delta)
        
        A = -torch.exp(self.A_log.float()) # (D, N)
        
        # Discretization
        # delta: (B, L, D)
        # A: (D, N)
        # B: (B, L, N)
        
        # Approximate: delta * A -> (B, L, D, N)
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B_ssm, x)
        
        # Scan (Sequential for PyTorch compatibility)
        h = torch.zeros((B, D, self.d_state), device=x.device)
        ys = []
        
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_u[:, t]
            y = torch.einsum('bdn,bn->bd', h, C_ssm[:, t])
            ys.append(y)
            
        y = torch.stack(ys, dim=1) # (B, L, D)
        return y + x * self.D

    def forward(self, x):
        # x: (B, L, D)
        
        residual = x
        x = self.norm(x)
        
        x_and_res = self.in_proj(x) # (B, L, 2 * d_inner)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        # Conv
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :residual.shape[1]]
        x = x.transpose(1, 2)
        
        x = self.act(x)
        
        y = self.ssm(x)
        
        y = y * self.act(res)
        
        out = self.out_proj(y)
        
        return out + residual

class RandomFeatureGaussianProcess(nn.Module):
    """
    Sparse Random Feature Gaussian Process (SNGP Head)
    Approximates GP using Random Fourier Features.
    """
    def __init__(self, in_features: int, out_features: int = 1, num_inducing: int = 1024):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_inducing = num_inducing
        
        # Random Feature Projection (Fixed)
        self.projection = nn.Linear(in_features, num_inducing, bias=False)
        self.projection.weight.requires_grad = False
        nn.init.normal_(self.projection.weight, mean=0.0, std=1.0) # Gaussian Kernel
        
        # Output Linear Layer (Learnable)
        # We model the mean prediction with this
        self.output_layer = nn.Linear(num_inducing, out_features)
        
        # Covariance Matrix (Ridge Regression update style or approximate)
        # For simplicity in Deep Learning, we often output the 'feature' 
        # and compute variance via Sigma = inv(Phi^T Phi + I) in complex implementations.
        # Here, we follow a simpler approximation: Var = phi * Sigma * phi^T
        # We will track Sigma during training or compute it batch-wise.
        # But for this simple implementation, we assume we just project to RFF 
        # and another head predicts variance, OR we use the logic of SNGP properly.
        
        # Proper SNGP requires updating precision matrix.
        # For this Phase 2 experiment, we will simplify: 
        # Mean = W * Phi(x)
        # Var = Phi(x) * Cov * Phi(x)^T
        # We'll treat Cov as a learnable parameter or identity for now 
        # but to adhere to SNGP, we usually update it.
        
        # Re-implementation for strict SNGP:
        # Since strict SNGP logic is complex, we will use a 'Laplace-like' head
        # where we return the hidden features (random features) to be used by
        # an exact GP or Laplace calc, OR we implement the MeanField approximation.
        
        # Alternative: We use a simplified RFF layer that transforms input 
        # and let the Variance tower predict standard variance, 
        # BUT SNGP specifically gets variance from distance in feature space.
        
        # Let's use the 'DUE' (Deterministic Uncertainty Estimation) approach or similar:
        # 1. Spectral Norm on encoder (handled in model)
        # 2. RFF Layer here.
        # 3. Variance = 1 - exp(- ||W phi(x)|| ) or similar.
        pass

    def forward(self, x):
        # x: (B, D)
        # Phi(x) = cos(Wx + b)
        p = self.projection(x)
        phi = torch.cos(p) * (2. / self.num_inducing)**0.5
        
        return self.output_layer(phi), phi

