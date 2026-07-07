
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Callable, Dict, Tuple
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP library not available. Install with: pip install shap")

class MetaSHAP:
    """
    Meta-SHAP: SHAP-based analysis of uncertainty sources
    Uses official SHAP package (GradientExplainer) for deep learning models.
    """
    def __init__(self, model: nn.Module, feature_names: List[str], X_background: np.ndarray = None, sid_background: np.ndarray = None):
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.X_background = X_background
        self.sid_background = sid_background
        
        # Determine device
        try:
             self.device = next(model.parameters()).device
        except StopIteration:
             self.device = torch.device('cpu')

        # Extract Station Embedding Layer for DynamicStationWrapper
        if hasattr(self.model, 'station_embedding') and self.model.station_embedding is not None:
             self.station_embedding_layer = self.model.station_embedding
        else:
             self.station_embedding_layer = None

    def _get_station_embeddings(self, station_ids: np.ndarray) -> torch.Tensor:
        """Helper to get continuous station embeddings from IDs"""
        if self.station_embedding_layer is None or station_ids is None:
            return None
        
        with torch.no_grad():
            sid_tensor = torch.from_numpy(station_ids).long().to(self.device)
            # Embedding layer forward
            embeddings = self.station_embedding_layer(sid_tensor)
        return embeddings

    def compute_all_shap_values(self, X: np.ndarray,
                                  target_func: Callable, # Used to determine mode
                                  station_ids: np.ndarray = None) -> np.ndarray:
        """
        Compute SHAP values for all samples (N, Seq, Feat) using GradientExplainer.
        Uses DynamicStationWrapper to handle station embeddings correctly.
        """
         # Determine mode based on target_func name (heuristic)
        mode = 'mean'
        if 'unc' in target_func.__name__.lower() or 'var' in target_func.__name__.lower() or 'width' in target_func.__name__.lower():
            mode = 'var'
            
        print(f"    [MetaSHAP] Explaining Output: {mode.upper()} (using Multi-Input GradientExplainer)")

        # 1. Prepare Background Data
        if self.X_background is None:
            print("    Warning: No background data provided. Using random subset of X as background.")
            bg_indices = np.random.choice(len(X), size=min(100, len(X)), replace=False)
            self.X_background = X[bg_indices]
            if station_ids is not None:
                self.sid_background = station_ids[bg_indices]
        
        # 2. Prepare Inputs (Main & Background)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        bg_X_tensor = torch.from_numpy(self.X_background).float().to(self.device)
        
        # Prepare Embeddings
        # We pass LIST of inputs to GradientExplainer: [X, Station_Embeddings]
        emb_tensor = self._get_station_embeddings(station_ids)
        bg_emb_tensor = self._get_station_embeddings(self.sid_background)
        
        # Dynamic Wrapper that accepts (X, Embeddings)
        class DynamicEmbedWrapper(nn.Module):
            def __init__(self, model, mode):
                super().__init__()
                self.model = model
                self.mode = mode
                
            def forward(self, x, emb=None):
                
                if hasattr(self.model, 'forward_with_embedding'):
                    mean, var = self.model.forward_with_embedding(x, emb)
                else:
                    # Fallback (Legacy/Fixed ID)
                    dummy_sid = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                    mean, var = self.model(x, dummy_sid)

                if self.mode == 'mean':
                    return mean
                return 2 * torch.sqrt(var) # Interval Width

        wrapped_model = DynamicEmbedWrapper(self.model, mode).to(self.device)
        
        # Prepare inputs list
        # If model has embeddings, inputs = [X, Emb]
        # If not, inputs = [X]
        if emb_tensor is not None and bg_emb_tensor is not None:
             inputs = [X_tensor, emb_tensor]
             bg_inputs = [bg_X_tensor, bg_emb_tensor]
        else:
             inputs = [X_tensor]
             bg_inputs = [bg_X_tensor]
             
        # Explainer
        explainer = shap.GradientExplainer(wrapped_model, bg_inputs)
        
        # Compute SHAP
        # Returns list of tensors (one per input) or single if one input?
        # GradientExplainer returns list/tensor corresponding to Inputs.
        shap_values = explainer.shap_values(inputs)
        
        # We only care about SHAP for X (first input)
        # shap_values is list of lists (Outputs x Inputs)? 
        # Or list of Inputs (if 1 output)?
        # iTransformer wrapper output is (N, 1) or (N,). Single output.
        # So shap_values is list of [SHAP_X, SHAP_Emb].
        if isinstance(shap_values, list):
            shap_values_x = shap_values[0] # The attribution to X
        else:
            shap_values_x = shap_values
            
        # Handle Output List (if wrapped in list again due to single output dimension)
        if isinstance(shap_values_x, list):
             shap_values_x = shap_values_x[0]

        if isinstance(shap_values_x, torch.Tensor):
            shap_values_x = shap_values_x.cpu().numpy()
            
        return shap_values_x # (N, Seq, Feat)

    def compute_global_importance(self, X: np.ndarray,
                                  target_func: Callable,
                                  station_ids: np.ndarray = None,
                                  n_permutations: int = 10) -> pd.DataFrame:
        """
        Compute Global Feature Importance: Mean(|SHAP|)
        """
        shap_values = self.compute_all_shap_values(X, target_func, station_ids)
        
        # Aggregate: Mean(|SHAP|) over N and Seq
        # shap_values: (N, Seq, Feat) or (N, Seq, Feat, 1)
        importances = np.mean(np.abs(shap_values), axis=(0, 1))
        
        if importances.ndim == 2 and importances.shape[-1] == 1:
            importances = importances.squeeze(-1) # (Feat,)
        
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })

    def compute_site_wise_importance(self, X: np.ndarray,
                                     target_func: Callable,
                                     station_ids: np.ndarray) -> pd.DataFrame:
        """
        Compute Site-wise Feature Importance
        Returns: DataFrame (Features x Sites)
        """
        shap_values = self.compute_all_shap_values(X, target_func, station_ids) # (N, Seq, Feat)
        
        # Group by Station ID
        # station_ids is (N,)
        unique_sites = np.unique(station_ids)
        site_importances = {}
        
        for site in unique_sites:
            idx = np.where(station_ids == site)[0]
            if len(idx) == 0: continue
            
            # Subset SHAP values for this site
            site_shap = shap_values[idx] # (n_site, Seq, Feat)
            
            # Aggregate: Mean(|SHAP|) over samples and time
            imp = np.mean(np.abs(site_shap), axis=(0, 1)) # (Feat,)
            
            # Map ID to Name
            # We need a map. Assuming we can get it or use raw ID.
            # For now use ID.
            site_importances[f'Site_{site}'] = imp
            
        df_site = pd.DataFrame(site_importances)
        df_site['Feature'] = self.feature_names
        return df_site

    def compute_high_uncertainty_importance(self, X: np.ndarray,
                                            target_func: Callable,
                                            station_ids: np.ndarray = None,
                                            percentile: int = 90) -> pd.DataFrame:
        """
        Compute Variable Importance for High Uncertainty Subset (Top 10%)
        """
        # 1. Compute Uncertainty (Interval Width)
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            sid_tensor = torch.from_numpy(station_ids).long().to(self.device) if station_ids is not None else None
            _, pred_var = self.model(X_tensor, sid_tensor)
            # Use interval width: 2 * sqrt(var)
            interval_width = 2 * np.sqrt(pred_var.cpu().numpy()).flatten()
            
        # 2. Filter Top Percentile
        threshold = np.percentile(interval_width, percentile)
        high_unc_idx = np.where(interval_width >= threshold)[0]
        
        if len(high_unc_idx) == 0:
            print("    Warning: No samples found for high uncertainty subset.")
            return None
            
        print(f"    [High Uncertainty] Analyzing top {100-percentile}% subset ({len(high_unc_idx)} samples)...")
        
        # 3. Compute SHAP for Subset
        X_subset = X[high_unc_idx]
        sid_subset = station_ids[high_unc_idx] if station_ids is not None else None
        
        # (N_sub, Seq, Feat)
        shap_values = self.compute_all_shap_values(X_subset, target_func, sid_subset)
        
        # 4. Aggregate: Mean(|SHAP|)
        # Mean over N and Seq
        imp = np.mean(np.abs(shap_values), axis=(0, 1))
        
        if imp.ndim == 2 and imp.shape[-1] == 1:
            imp = imp.squeeze(-1)
        
        df = pd.DataFrame({'Feature': self.feature_names, 'Importance_HighUncertainty': imp})
        return df.sort_values('Importance_HighUncertainty', ascending=False)

    def compute_temporal_importance(self, X: np.ndarray,
                                    target_func: Callable,
                                    station_ids: np.ndarray = None,
                                    uncertainty_mode: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Compute Temporal Lag Importance (Features x Seq_Len)
        
        Args:
            uncertainty_mode: if True, also returns 'High_Uncertainty' heatmap 
                              (subset of top 10% uncertainty samples)
        """
        shap_values = self.compute_all_shap_values(X, target_func, station_ids) # (N, Seq, Feat)
        
        # 1. General Temporal Importance (Mean(|SHAP|) over N)
        # Result: (Seq, Feat)
        temp_imp = np.mean(np.abs(shap_values), axis=0)
        
        if temp_imp.ndim == 3 and temp_imp.shape[-1] == 1:
            temp_imp = temp_imp.squeeze(-1) # (Seq, Feat)
        df_temp = pd.DataFrame(temp_imp, columns=self.feature_names)
        df_temp.index.name = 'Lag'
        
        result = {'general': df_temp}
        
        # 2. High Uncertainty Focus
        if uncertainty_mode:
            # We need to compute uncertainty (interval width) for all samples to filter
            # But the caller (run_full_pipeline) might have it?
            # Or we re-compute.
            with torch.no_grad():
                # For filter, we just need prediction, not explanation.
                X_tensor = torch.from_numpy(X).float().to(self.device)
                sid_tensor = torch.from_numpy(station_ids).long().to(self.device) if station_ids is not None else None
                _, pred_var = self.model(X_tensor, sid_tensor)
                interval_width = 2 * np.sqrt(pred_var.cpu().numpy()).flatten()
                
            # Filter Top 10%
            threshold = np.percentile(interval_width, 90)
            high_unc_idx = np.where(interval_width >= threshold)[0]
            
            if len(high_unc_idx) > 0:
                print(f"    [Temporal] Analyzing High Uncertainty Subset (Top 10%: {len(high_unc_idx)} samples)")
                high_unc_shap = shap_values[high_unc_idx] # (n_subset, Seq, Feat)
                temp_imp_high = np.mean(np.abs(high_unc_shap), axis=0)
                
                if temp_imp_high.ndim == 3 and temp_imp_high.shape[-1] == 1:
                    temp_imp_high = temp_imp_high.squeeze(-1)
                    
                df_temp_high = pd.DataFrame(temp_imp_high, columns=self.feature_names)
                df_temp_high.index.name = 'Lag'
                result['high_uncertainty'] = df_temp_high
                
        return result

    def compute_site_importance(self, X: np.ndarray, target_func: Callable, 
                              station_ids: np.ndarray = None,
                              feature_names: List[str] = None) -> pd.DataFrame:
        """
        Compute site-specific feature importance
        """
        if station_ids is None:
            print("Warning: No station_ids provided for site importance.")
            return None
            
        shap_values = self.compute_all_shap_values(X, target_func, station_ids)
        
        unique_sites = np.unique(station_ids)
        # TODO: Move site map to constants or pass as arg
        site_id_to_name = {0: '공주', 1: '대청호', 2: '갑천', 3: '부여', 4: '용담호'}
        
        # If feature_names provided, use them for validation/filtering
        if feature_names is None:
            feature_names = self.feature_names
            
        site_data = {'Feature': feature_names}
        
        for site_id in [0, 1, 2, 3, 4]: # Ensure all 5 sites are covered
            indices = np.where(station_ids == site_id)[0]
            site_name = site_id_to_name.get(site_id, f"Site_{site_id}")
            
            if len(indices) == 0:
                site_data[site_name] = [0.0] * len(feature_names)
                continue
                
            site_shap = shap_values[indices]  # (n_site, Seq, Feat)
            
            # Mean absolute SHAP over samples and time
            importance = np.mean(np.abs(site_shap), axis=(0, 1))
            site_data[site_name] = importance.flatten()
            
        return pd.DataFrame(site_data)

    def compute_prediction_importance(self, X: np.ndarray, target_func_pred: Callable, station_ids: np.ndarray = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Wrapper for Global Importance (Prediction) to match pipeline calls"""
        importance = self.compute_global_importance(X, target_func_pred, station_ids)
        # Dummy p-values for now to satisfy unpacking, could call compute_global_p_values later
        p_values = pd.DataFrame({'Feature': self.feature_names, 'P_Value_T_Test': [1.0]*len(self.feature_names)})
        return importance, p_values

    def compute_interval_width_importance(self, X: np.ndarray, target_func_unc: Callable, station_ids: np.ndarray = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Wrapper for Global Importance (Uncertainty) to match pipeline calls"""
        importance = self.compute_global_importance(X, target_func_unc, station_ids)
        p_values = pd.DataFrame({'Feature': self.feature_names, 'P_Value_T_Test': [1.0]*len(self.feature_names)})
        return importance, p_values

    def get_beeswarm_data(self, X: np.ndarray, target_func: Callable, station_ids: np.ndarray = None) -> pd.DataFrame:
        """Beeswarm 플롯을 위한 SHAP 및 Feature Value 쌍 생성"""
        shap_values = self.compute_all_shap_values(X, target_func, station_ids) # (N, Seq, Feat)
        
        # 마지막 타임스텝 기준 데이터 추출 (가장 최신 영향력)
        last_shap = shap_values[:, -1, :] # (N, Feat)
        last_X = X[:, -1, :] # (N, Feat)
        
        data = []
        for f_idx, f_name in enumerate(self.feature_names):
            df_feat = pd.DataFrame({
                'Feature': f_name,
                'SHAP_Value': last_shap[:, f_idx].flatten(),
                'Value': last_X[:, f_idx].flatten()
            })
            data.append(df_feat)
            
        return pd.concat(data, ignore_index=True)
