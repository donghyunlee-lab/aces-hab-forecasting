
from typing import Dict
from src.models.forecasting import iTransformer, GRUModel, PatchTST, TCNModel, MambaModel, MambaSNGP, MambaDER, MambaISODER

def get_model(config: Dict, n_features: int, n_stations: int):
    """Factory function to create model based on config"""
    model_type = config.get('model_type', 'iTransformer')
    
    common_args = {
        'seq_len': config['seq_len'],
        'n_features': n_features,
        'hidden_dim': config['hidden_dim'],
        'n_layers': config['n_layers'],
        'dropout_rate': config['dropout_rate'],
        'n_stations': n_stations,
        'use_station_embedding': config.get('use_panel_data', True),
        'use_station_embedding': config.get('use_panel_data', True),
        'decoupled': True # Force Decoupled for Ablation Study
    }
    
    if model_type == 'iTransformer':
        return iTransformer(n_heads=config['n_heads'], **common_args)
    elif model_type == 'GRU':
        return GRUModel(**common_args)
    elif model_type == 'PatchTST':
        return PatchTST(n_heads=config['n_heads'], **common_args)
    elif model_type == 'TCN':
        return TCNModel(**common_args)
    elif model_type == 'Mamba':
        return MambaModel(**common_args)
    elif model_type == 'MambaSNGP':
        return MambaSNGP(**common_args)
    elif model_type == 'MambaDER':
        return MambaDER(**common_args)
    elif model_type == 'MambaISODER':
        return MambaISODER(**common_args)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

