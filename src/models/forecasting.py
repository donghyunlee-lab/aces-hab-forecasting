
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from src.models.layers import TemporalBlock

# ============================================================================
# iTRANSFORMER WITH INVERTED ATTENTION
# ============================================================================
class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformer for Time Series with Station Embedding

    혁신: 시간(T) 차원 대신 변수(D) 차원에 Attention을 적용
    iTransformer with Phase 2 Uncertainty Head
    Supports both Shared (Baseline) and Decoupled (Adaptive-Detach) Architectures.
    """
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout_rate: float = 0.3,
                 n_stations: int = 5, use_station_embedding: bool = True,
                 decoupled: bool = True): # Default to True for backward compatibility
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.use_station_embedding = use_station_embedding
        self.decoupled = decoupled
        
        if use_station_embedding:
            self.station_embedding = nn.Embedding(n_stations, hidden_dim // 4)
        else:
            self.station_embedding = None

        # Common Architecture Blocks
        def create_cnn():
            return nn.Sequential(
                nn.Conv1d(in_channels=n_features, out_channels=hidden_dim // 2, 
                         kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Conv1d(in_channels=hidden_dim // 2, out_channels=n_features,
                         kernel_size=3, padding=1),
                nn.BatchNorm1d(n_features),
                nn.ReLU()
            )
            
        def create_transformer():
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout_rate, batch_first=True, activation='gelu'
            )
            return nn.TransformerEncoder(layer, num_layers=n_layers)

        self.embedding_dim = hidden_dim + (hidden_dim // 4 if use_station_embedding else 0)
        
        def create_processor():
            return nn.Sequential(
                nn.Linear(self.embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )

        # 1. Encoders
        if self.decoupled:
            # --- Decoupled Architecture (Separate Towers) ---
            self.temporal_cnn_mean = create_cnn()
            self.var_input_proj = nn.Linear(n_features, n_features) # Unique to Var tower
            self.temporal_cnn_var = create_cnn()
            
            self.feature_embedding_mean = nn.Linear(seq_len, hidden_dim)
            self.feature_embedding_var = nn.Linear(seq_len, hidden_dim)
            
            self.itransformer_mean = create_transformer()
            self.itransformer_var = create_transformer()
            
            self.mean_temporal_processor = create_processor()
            self.var_temporal_processor = create_processor()
            
        else:
            # --- Shared Architecture (Single Tower) ---
            self.temporal_cnn_shared = create_cnn()
            self.feature_embedding_shared = nn.Linear(seq_len, hidden_dim)
            self.itransformer_shared = create_transformer()
            self.shared_processor = create_processor()

        # 2. Heads (Always separate)
        self.mean_predictor = nn.Linear(hidden_dim // 2, 1)
        self.var_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()
        )
        
        with torch.no_grad():
            self.var_predictor[-2].bias.fill_(1.0)
            
        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor, station_ids: torch.Tensor = None, mc_samples: int = 1):
        # Handle Station Embedding
        station_emb = None
        if self.use_station_embedding and station_ids is not None:
            station_emb = self.station_embedding(station_ids) # (batch, hidden//4)
            
        return self.forward_with_embedding(x, station_emb)

    def forward_with_embedding(self, x: torch.Tensor, station_embedding: torch.Tensor = None):
        """
        Unified forward pass handling both Shared and Decoupled logic.
        """
        if self.decoupled:
            # --- Decoupled Path ---
            # Mean Path
            x_cnn_m = self.temporal_cnn_mean(x.transpose(1, 2))
            x_enh_m = x.transpose(1, 2) + x_cnn_m
            x_emb_m = self.feature_embedding_mean(x_enh_m)
            x_trans_m = self.itransformer_mean(x_emb_m)
            x_pool_m = x_trans_m.mean(dim=1)
            
            # Var Path
            x_var_proj = self.var_input_proj(x)
            x_cnn_v = self.temporal_cnn_var(x_var_proj.transpose(1, 2))
            x_enh_v = x_var_proj.transpose(1, 2) + x_cnn_v
            x_emb_v = self.feature_embedding_var(x_enh_v)
            x_trans_v = self.itransformer_var(x_emb_v)
            x_pool_v = x_trans_v.mean(dim=1)
            
            if station_embedding is not None:
                x_pool_m = torch.cat([x_pool_m, station_embedding], dim=1)
                x_pool_v = torch.cat([x_pool_v, station_embedding], dim=1)
                
            x_proc_m = self.mean_temporal_processor(x_pool_m)
            x_proc_v = self.var_temporal_processor(x_pool_v)
            
            mean = self.mean_predictor(x_proc_m)
            var = self.var_predictor(x_proc_v) + 1e-6
            
        else:
            # --- Shared Path ---
            x_cnn = self.temporal_cnn_shared(x.transpose(1, 2))
            x_enh = x.transpose(1, 2) + x_cnn
            x_emb = self.feature_embedding_shared(x_enh)
            x_trans = self.itransformer_shared(x_emb)
            x_pool = x_trans.mean(dim=1)
            
            if station_embedding is not None:
                x_pool = torch.cat([x_pool, station_embedding], dim=1)
                
            x_proc = self.shared_processor(x_pool)
            
            mean = self.mean_predictor(x_proc)
            var = self.var_predictor(x_proc) + 1e-6
            
        return mean, var


class TCNModel(nn.Module):
    """
    TCN for Time Series Forecasting
    Dilated Causal Convolutions + Station Embedding + Gaussian NLL Head
    Supports both Shared and Decoupled architectures
    """
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 64,
                 n_layers: int = 2, dropout_rate: float = 0.3,
                 n_stations: int = 5, use_station_embedding: bool = True,
                 kernel_size: int = 3, decoupled: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.use_station_embedding = use_station_embedding
        self.decoupled = decoupled
        
        # Station Embedding
        if use_station_embedding:
            self.station_embedding = nn.Embedding(n_stations, hidden_dim // 4)
            input_dim = n_features + (hidden_dim // 4)
        else:
            input_dim = n_features
        
        def create_tcn():
            layers = []
            num_levels = n_layers
            num_channels = [hidden_dim] * num_levels
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_channels = input_dim if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                        dilation=dilation_size,
                                        padding=(kernel_size-1) * dilation_size, dropout=dropout_rate)]
            return nn.Sequential(*layers)
        
        if self.decoupled:
            self.tcn_mean = create_tcn()
            self.tcn_var = create_tcn()
        else:
            self.tcn = create_tcn()
        
        # Gaussian NLL Head
        self.mean_predictor = nn.Linear(hidden_dim, 1)
        self.var_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        with torch.no_grad():
            self.var_predictor[-2].bias.fill_(1.0)
            
    def forward(self, x: torch.Tensor, station_ids: torch.Tensor = None, mc_samples: int = 1):
        # x: (batch, seq_len, features)
        
        # Station Embedding Integration (Concatenation)
        if self.use_station_embedding and station_ids is not None:
            station_emb = self.station_embedding(station_ids) # (batch, emb_dim)
            station_emb_seq = station_emb.unsqueeze(1).repeat(1, self.seq_len, 1) # (batch, seq, emb_dim)
            x = torch.cat([x, station_emb_seq], dim=2)
            
        # Transpose for TCN: (batch, features, seq_len)
        x_tcn = x.transpose(1, 2)
        
        if self.decoupled:
            # Mean Tower
            y_mean = self.tcn_mean(x_tcn)
            last_hidden_mean = y_mean[:, :, -1]
            mean = self.mean_predictor(last_hidden_mean)
            
            # Var Tower
            y_var = self.tcn_var(x_tcn)
            last_hidden_var = y_var[:, :, -1]
            var = self.var_predictor(last_hidden_var) + 1e-6
        else:
            y = self.tcn(x_tcn) # (batch, hidden, seq)
            last_hidden = y[:, :, -1] # (batch, hidden)
            mean = self.mean_predictor(last_hidden)
            var = self.var_predictor(last_hidden) + 1e-6
            
        return mean, var

class GRUModel(nn.Module):
    """
    GRU for Time Series Forecasting
    Standard GRU + Gaussian NLL Output Head
    Optionally separates Mean and Variance Encoders (Adaptive-Detach)
    """
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 64,
                 n_layers: int = 2, dropout_rate: float = 0.3,
                 n_stations: int = 5, use_station_embedding: bool = True,
                 decoupled: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.use_station_embedding = use_station_embedding
        self.decoupled = decoupled
        
        if use_station_embedding:
            self.station_embedding = nn.Embedding(n_stations, hidden_dim // 4)
            input_dim = n_features + (hidden_dim // 4)
        else:
            input_dim = n_features

        # Define GRU Encoder(s)
        gru_config = dict(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0
        )
        
        if self.decoupled:
            self.gru_mean = nn.GRU(**gru_config)
            self.gru_var = nn.GRU(**gru_config)
        else:
            self.gru = nn.GRU(**gru_config)
            
        # Gaussian NLL Head
        self.mean_predictor = nn.Linear(hidden_dim, 1)
        self.var_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Init variance bias
        with torch.no_grad():
            self.var_predictor[-2].bias.fill_(1.0)

    def forward(self, x: torch.Tensor, station_ids: torch.Tensor = None, mc_samples: int = 1):
        """
        Forward pass for BayesianGRU
        x: (batch, seq_len, features)
        station_ids: (batch,) - Station IDs for embedding lookup
        """
        # Station Embedding Integration
        if self.use_station_embedding and station_ids is not None:
            station_emb = self.station_embedding(station_ids)  # (batch, emb_dim)
            station_emb_seq = station_emb.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq, emb_dim)
            x = torch.cat([x, station_emb_seq], dim=2)
            
        if self.decoupled:
            # Mean Tower
            out_mean, _ = self.gru_mean(x)
            last_hidden_mean = out_mean[:, -1, :]
            mean = self.mean_predictor(last_hidden_mean)
            
            # Variance Tower
            out_var, _ = self.gru_var(x)
            last_hidden_var = out_var[:, -1, :]
            var = self.var_predictor(last_hidden_var) + 1e-6
        else:
            # Shared Tower
            out, _ = self.gru(x)  # (batch, seq, hidden)
            last_hidden = out[:, -1, :]  # (batch, hidden)
            
            mean = self.mean_predictor(last_hidden)
            var = self.var_predictor(last_hidden) + 1e-6
            
        return mean, var

    def forward_with_embedding(self, x: torch.Tensor, station_emb: torch.Tensor):
        """
        Forward pass with explicit embedding injection for SHAP gradient calculation.
        x: (batch, seq_len, features)
        station_emb: (batch, emb_dim) - Dense embedding tensor
        """
        if self.use_station_embedding and station_emb is not None:
            # Expand provided embedding to sequence length
            station_emb_seq = station_emb.unsqueeze(1).repeat(1, self.seq_len, 1) # (batch, seq, emb_dim)
            x = torch.cat([x, station_emb_seq], dim=2)
        
        if self.decoupled:
            # Mean Tower
            out_mean, _ = self.gru_mean(x)
            last_hidden_mean = out_mean[:, -1, :]
            mean = self.mean_predictor(last_hidden_mean)
            
            # Variance Tower
            out_var, _ = self.gru_var(x)
            last_hidden_var = out_var[:, -1, :]
            var = self.var_predictor(last_hidden_var) + 1e-6
        else:
            out, _ = self.gru(x)
            last_hidden = out[:, -1, :]
            
            mean = self.mean_predictor(last_hidden)
            var = self.var_predictor(last_hidden) + 1e-6
            
        return mean, var


class PatchTST(nn.Module):
    """
    PatchTST (simplified) with Gaussian NLL Head
    Key: Patching + Transformer Backbone + Flatten Head
    Supports both Shared and Decoupled architectures
    """
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout_rate: float = 0.3,
                 n_stations: int = 5, use_station_embedding: bool = True,
                 patch_len: int = 10, stride: int = 5, decoupled: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        self.decoupled = decoupled
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        if use_station_embedding:
            self.station_embedding = nn.Embedding(n_stations, hidden_dim)
        else:
            self.station_embedding = None

        def create_transformer():
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout_rate,
                batch_first=True,
                activation='gelu'
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head_dim = self.num_patches * hidden_dim
        
        def create_readout():
            return nn.Sequential(
                nn.Linear(self.head_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        
        if self.decoupled:
            self.patch_embedding_mean = nn.Linear(patch_len, hidden_dim)
            self.patch_embedding_var = nn.Linear(patch_len, hidden_dim)
            self.transformer_mean = create_transformer()
            self.transformer_var = create_transformer()
            self.readout_mean = create_readout()
            self.readout_var = create_readout()
        else:
            self.patch_embedding = nn.Linear(patch_len, hidden_dim)
            self.transformer = create_transformer()
            self.readout = create_readout()
        
        self.mean_predictor = nn.Linear(hidden_dim, 1)
        self.var_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        with torch.no_grad():
            self.var_predictor[-2].bias.fill_(1.0)

    def forward(self, x: torch.Tensor, station_ids: torch.Tensor = None, mc_samples: int = 1):
        return self.forward_with_embedding(x, station_ids)

    def forward_with_embedding(self, x: torch.Tensor, station_ids_or_emb: torch.Tensor = None):
        batch_size, seq_len, n_vars = x.shape
        
        # Patching
        x_unfold = x.permute(0, 2, 1).unfold(dimension=2, size=self.patch_len, step=self.stride)
        x_patches = x_unfold.reshape(batch_size * n_vars, self.num_patches, self.patch_len)
        
        # Get station embedding
        station_emb = None
        if self.station_embedding is not None and station_ids_or_emb is not None:
            if station_ids_or_emb.dtype == torch.long:
                station_emb = self.station_embedding(station_ids_or_emb)
            else:
                station_emb = station_ids_or_emb
            station_emb = station_emb.repeat_interleave(n_vars, dim=0).unsqueeze(1)
        
        if self.decoupled:
            # Mean Tower
            enc_mean = self.patch_embedding_mean(x_patches)
            if station_emb is not None:
                enc_mean = enc_mean + station_emb
            enc_mean = self.transformer_mean(enc_mean)
            enc_mean = enc_mean.reshape(batch_size, n_vars, -1)
            out_mean = self.readout_mean(enc_mean).mean(dim=1)
            mean = self.mean_predictor(out_mean)
            
            # Var Tower
            enc_var = self.patch_embedding_var(x_patches)
            if station_emb is not None:
                enc_var = enc_var + station_emb
            enc_var = self.transformer_var(enc_var)
            enc_var = enc_var.reshape(batch_size, n_vars, -1)
            out_var = self.readout_var(enc_var).mean(dim=1)
            var = self.var_predictor(out_var) + 1e-6
        else:
            enc_out = self.patch_embedding(x_patches)
            if station_emb is not None:
                enc_out = enc_out + station_emb
            enc_out = self.transformer(enc_out)
            enc_out = enc_out.reshape(batch_size, n_vars, -1)
            out = self.readout(enc_out).mean(dim=1)
            mean = self.mean_predictor(out)
            var = self.var_predictor(out) + 1e-6
            
        return mean, var


class MambaModel(nn.Module):
    """
    Mamba for Time Series Forecasting
    Pure PyTorch Implementation of Mamba Backbone
    Supports Decoupled Architecture (Mean/Var Towers)
    """
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 64,
                 n_layers: int = 2, dropout_rate: float = 0.0,
                 n_stations: int = 5, use_station_embedding: bool = True,
                 decoupled: bool = True):
        super().__init__()
        from src.models.layers import MambaBlock
        
        self.seq_len = seq_len
        self.use_station_embedding = use_station_embedding
        self.decoupled = decoupled
        
        if use_station_embedding:
            self.station_embedding = nn.Embedding(n_stations, hidden_dim // 4)
            input_dim = n_features + (hidden_dim // 4)
        else:
            input_dim = n_features
            
        # Input Projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        def create_mamba_stack():
            layers = []
            for _ in range(n_layers):
                layers.append(MambaBlock(d_model=hidden_dim))
            return nn.Sequential(*layers)
            
        if self.decoupled:
            self.mamba_mean = create_mamba_stack()
            self.mamba_var = create_mamba_stack()
        else:
            self.mamba = create_mamba_stack()
            
        # Heads
        self.mean_predictor = nn.Linear(hidden_dim, 1)
        self.var_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        with torch.no_grad():
            self.var_predictor[-2].bias.fill_(1.0)

    def forward(self, x: torch.Tensor, station_ids: torch.Tensor = None, mc_samples: int = 1):
        # x: (batch, seq_len, features)

        # Station Embedding
        if self.use_station_embedding and station_ids is not None:
            station_emb = self.station_embedding(station_ids) # (batch, emb_dim)
        else:
            station_emb = None
        return self.forward_with_embedding(x, station_emb)

    def forward_with_embedding(self, x: torch.Tensor, station_emb: torch.Tensor = None):
        if self.use_station_embedding and station_emb is not None:
            station_emb_seq = station_emb.unsqueeze(1).repeat(1, self.seq_len, 1)
            x = torch.cat([x, station_emb_seq], dim=2)

        x = self.input_proj(x)

        if self.decoupled:
            x_mean = self.mamba_mean(x)
            last_mean = x_mean[:, -1, :]
            mean = self.mean_predictor(last_mean)

            x_var = self.mamba_var(x)
            last_var = x_var[:, -1, :]
            var = self.var_predictor(last_var) + 1e-6
        else:
            x_out = self.mamba(x)
            last_out = x_out[:, -1, :]
            mean = self.mean_predictor(last_out)
            var = self.var_predictor(last_out) + 1e-6

        return mean, var


class MambaSNGP(MambaModel):
    """
    Mamba with Spectral Normalization and Gaussian Process (SNGP)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from torch.nn.utils import spectral_norm
        from src.models.layers import RandomFeatureGaussianProcess
        
        # Apply Spectral Norm to all Linear/Conv layers in Mamba Backbone
        def apply_sn(module):
            for name, layer in module.named_children():
                if isinstance(layer, (nn.Linear, nn.Conv1d)):
                    spectral_norm(layer)
                else:
                    apply_sn(layer)
                    
        if self.decoupled:
            apply_sn(self.mamba_mean)
            apply_sn(self.mamba_var)
        else:
            apply_sn(self.mamba)
            
        hidden_dim = kwargs.get('hidden_dim', 64)
        self.gp_head = RandomFeatureGaussianProcess(hidden_dim, out_features=1)
        
    def forward(self, x: torch.Tensor, station_ids: torch.Tensor = None, mc_samples: int = 1):
        if self.use_station_embedding and station_ids is not None:
            station_emb = self.station_embedding(station_ids)
            station_emb_seq = station_emb.unsqueeze(1).repeat(1, self.seq_len, 1)
            x = torch.cat([x, station_emb_seq], dim=2)
            
        x = self.input_proj(x)
        
        if self.decoupled:
            # Mean Tower (Standard with SN)
            x_mean = self.mamba_mean(x)
            last_mean = x_mean[:, -1, :]
            mean = self.mean_predictor(last_mean)
            
            # Var Tower (GP Head)
            x_var = self.mamba_var(x)
            last_var_feat = x_var[:, -1, :] 
            
            # GP Head returns (pred, phi)
            gp_out, _ = self.gp_head(last_var_feat)
            var = nn.functional.softplus(gp_out) + 1e-6
            
        else:
            x_out = self.mamba(x)
            last_out = x_out[:, -1, :] 
            mean = self.mean_predictor(last_out)
            gp_out, _ = self.gp_head(last_out)
            var = nn.functional.softplus(gp_out) + 1e-6
            
        return mean, var


class MambaDER(MambaModel):
    """
    Mamba with Deep Evidential Regression Head
    Outputs: Gamma, Nu, Alpha, Beta
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = kwargs.get('hidden_dim', 64)
        
        # We keep mean_predictor for Gamma
        # We replace var_predictor for Nu, Alpha, Beta
        self.uncertainty_predictor = nn.Linear(hidden_dim, 3) # [nu, alpha, beta]
        
        # Initialize biases for stability
        with torch.no_grad():
            self.uncertainty_predictor.bias[1].fill_(3.0) # alpha > 1
            self.uncertainty_predictor.bias[2].fill_(1.0) # beta initial value
        
    def get_evidential_params(self, x: torch.Tensor, station_ids: torch.Tensor = None):
        if self.use_station_embedding and station_ids is not None:
            station_emb = self.station_embedding(station_ids)
            station_emb_seq = station_emb.unsqueeze(1).repeat(1, self.seq_len, 1)
            x = torch.cat([x, station_emb_seq], dim=2)
        x = self.input_proj(x)
        
        if self.decoupled:
            x_mean = self.mamba_mean(x)[:, -1, :]
            gamma = self.mean_predictor(x_mean)
            
            x_var = self.mamba_var(x)[:, -1, :]
            unc_params = self.uncertainty_predictor(x_var)
            unc_params = nn.functional.softplus(unc_params) + 1e-6
            nu = unc_params[:, 0:1]
            alpha = unc_params[:, 1:2] + 1.0 # Ensure alpha > 1
            beta = unc_params[:, 2:3]
        else:
            x_out = self.mamba(x)[:, -1, :]
            gamma = self.mean_predictor(x_out)
            unc_params = self.uncertainty_predictor(x_out)
            unc_params = nn.functional.softplus(unc_params) + 1e-6
            nu = unc_params[:, 0:1]
            alpha = unc_params[:, 1:2] + 1.0 # Ensure alpha > 1
            beta = unc_params[:, 2:3]
            
        return gamma, nu, alpha, beta

    def forward(self, x: torch.Tensor, station_ids: torch.Tensor = None, mc_samples: int = 1):
        gamma, nu, alpha, beta = self.get_evidential_params(x, station_ids)
        
        # Calculate Total Variance for ADNLL Loss compatibility
        # Var = Aleatoric + Epistemic
        # Aleatoric = beta / (alpha - 1)
        # Epistemic = beta / (nu * (alpha - 1))
        
        from src.evaluation.uncertainty import compute_der_uncertainty
        aleatoric, epistemic = compute_der_uncertainty(gamma, nu, alpha, beta)
        
        return gamma, aleatoric + epistemic


class MambaISODER(MambaDER):
    """
    Mamba with ISO-DER (DER + ISO Penalties)
    Designed for Phase 3 Calibration & Optimization.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store ISO params
        self.lambd_indep = kwargs.get('lambd_indep', 0.1)
        self.lambd_sharp = kwargs.get('lambd_sharp', 0.05)
        
        # ISO-DER Specific Initialization
        # We want Alpha >> 1 initially to stabilize Aleatoric Variance (Beta/(Alpha-1))
        # Beta should be reasonable (e.g., 1.0)
        # Gamma (Mean) is handled by main network.
        # Nu (Evidence count) can start small.
        with torch.no_grad():
            self.uncertainty_predictor.bias[1].fill_(3.0) # alpha init: Softplus(3.0) ~ 3.05 (Safe > 1)
            self.uncertainty_predictor.bias[2].fill_(10.0) # beta init: 10.0 -> Broad initial variance
            
    def get_evidential_params(self, x: torch.Tensor, station_ids: torch.Tensor = None):
        """
        Return RAW parameters (or standard activated) for ISO-DER Loss.
        Note: iso_der_loss applies additional safeguards (Softplus+1.1 for Alpha).
        We return standard Softplus outputs here to maintain compatibility with MambaDER structure,
        but Loss will further constrain them.
        """
        return super().get_evidential_params(x, station_ids)


