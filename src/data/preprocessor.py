
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """
    Phase 1: Panel Data Preprocessor for Multi-Site HAB Prediction
    
    핵심 기능:
    1. 5개 사이트 데이터 통합 (공주, 대청호, 갑천, 부여, 용담호)
    2. Station_ID 추가 (0-4) 및 사이트별 특성 학습
    3. 사이트별 시퀀스 생성 (temporal leakage 방지)
    """

    CORE_FEATURES = [
        '수온 (℃)', '수소이온농도', '전기전도도 (μS/cm)', '용존산소 (mg/L)',
        '탁도 (NTU)', '총유기탄소 (mg/L)', '클로로필-a (mg/㎥)',
        '총질소 (mg/L)', '총인 (mg/L)'
    ]

    def __init__(self, data_path: str, site_names: List[str] = None,
                 apply_log_transform: bool = True, use_panel_data: bool = True,
                 core_features_only: bool = True):
        self.data_path = data_path
        # Phase 1: Panel data mode - multiple sites
        if site_names is None:
            site_names = ['공주', '대청호', '갑천', '부여', '용담호']  # 5개 사이트
        self.site_names = site_names
        self.use_panel_data = use_panel_data
        self.scaler = MinMaxScaler()
        self.feature_names = None
        self.apply_log_transform = apply_log_transform
        self.core_features_only = core_features_only
        self.log_transform_cols = []  # 로그 변환된 컬럼 추적
        self.target_idx = None  # 타겟 변수 인덱스 (Chl-a)
        self.station_id_map = {site: idx for idx, site in enumerate(site_names)}  # 사이트 → ID 매핑
        self.site_id_map_rev = {v: k for k, v in self.station_id_map.items()} # ID → 사이트 매핑
        print(f"Phase 1: Panel Data Mode - {len(site_names)} sites")
        print(f"  Station ID mapping: {self.station_id_map}")

    def load_data(self) -> pd.DataFrame:
        """
        Phase 1: Load and stack data from multiple sites
        
        Returns:
            panel_df: Combined dataframe with Station_ID column
        """
        df = pd.read_csv(self.data_path)
        df['측정일'] = pd.to_datetime(df['측정일'])

        if self.use_panel_data:
            # Filter for target sites and stack
            panel_dfs = []
            for site_name in self.site_names:
                site_df = df[df['측정소'] == site_name].copy()
                if len(site_df) > 0:
                    site_df['Station_ID'] = self.station_id_map[site_name]
                    site_df = site_df.sort_values('측정일').reset_index(drop=True)
                    panel_dfs.append(site_df)
                    print(f"  {site_name} (ID={self.station_id_map[site_name]}): {len(site_df)} records")
            
            if len(panel_dfs) == 0:
                raise ValueError(f"No data found for sites: {self.site_names}")
            
            # Stack all sites (no temporal overlap/leakage)
            panel_df = pd.concat(panel_dfs, ignore_index=True)
            panel_df = panel_df.sort_values(['Station_ID', '측정일']).reset_index(drop=True)
            
            print(f"\nPanel Data Summary:")
            print(f"  Total records: {len(panel_df)}")
            print(f"  Date range: {panel_df['측정일'].min()} to {panel_df['측정일'].max()}")
            print(f"  Sites: {panel_df['Station_ID'].value_counts().sort_index().to_dict()}")
            
            return panel_df
        else:
            # Single site mode (backward compatibility)
            if len(self.site_names) > 0:
                site_name = self.site_names[0]
            else:
                site_name = '공주'
            site_df = df[df['측정소'] == site_name].copy()
            site_df = site_df.sort_values('측정일').reset_index(drop=True)
            print(f"Loaded {len(site_df)} records for site: {site_name}")
            return site_df

    def preprocess(self, df: pd.DataFrame,
                   fit_period: Tuple[str, str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess data:
        - Select relevant features (numeric columns)
        - Apply log transformation to target and high-variance features
        - Add feature engineering (1-day diff, 7-day moving average)
        - Handle missing values (already imputed)
        - Normalize features
        """
        # Define features to use (exclude metadata columns)
        exclude_cols = ['수계', '측정소', '측정일', '시작연도', '종료연도', '파일명', '측정소명']
        if self.use_panel_data:
            exclude_cols.append('Station_ID')  # Phase 1: Station_ID는 별도로 관리
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64]]

        if fit_period is None:
            raise ValueError("fit_period is required: preprocessing statistics must be fit on training data only")
        fit_start, fit_end = pd.to_datetime(fit_period[0]), pd.to_datetime(fit_period[1])
        fit_mask = (df['측정일'] >= fit_start) & (df['측정일'] <= fit_end)
        if not fit_mask.any():
            raise ValueError(f"fit_period {fit_period} contains no rows")

        # The redesign excludes the highly imputed trace-organic columns.  They
        # lack a direct Chl-a mechanism and add avoidable reconstruction risk.
        if self.core_features_only:
            missing_core = [col for col in self.CORE_FEATURES if col not in feature_cols]
            if missing_core:
                raise ValueError(f"missing required core features: {missing_core}")
            feature_cols = list(self.CORE_FEATURES)

        # Exclude ECD measurement variables (keep only general measurement)
        # ECD variables are duplicates of the same substances measured with different methods
        ecd_vars = [col for col in feature_cols if '[ECD]' in col]
        if len(ecd_vars) > 0:
            print(f"  Excluding ECD measurement variables ({len(ecd_vars)} variables): {ecd_vars}")
            feature_cols = [col for col in feature_cols if '[ECD]' not in col]
            print(f"  Remaining features after ECD exclusion: {len(feature_cols)}")

        # Find target variable (Chl-a)
        target_col = None
        for col in feature_cols:
            if '클로로필' in col or 'Chlorophyll' in col or 'Chl' in col:
                target_col = col
                break
        
        if target_col:
            self.target_idx = feature_cols.index(target_col)
            print(f"Target variable identified: {target_col} (index: {self.target_idx})")

        # Copy for processing
        df_numeric = df[feature_cols].copy()
        
        # Log transformation for target and high-variance features
        if self.apply_log_transform:
            # Identify high-variance features (coefficient of variation > 1.0)
            high_var_cols = []
            train_numeric = df_numeric.loc[fit_mask]
            for col in feature_cols:
                if train_numeric[col].std() > 0:
                    cv = train_numeric[col].std() / (train_numeric[col].mean() + 1e-6)
                    if cv > 1.0 or col == target_col:
                        high_var_cols.append(col)
            
            # Apply log1p transformation
            for col in high_var_cols:
                if train_numeric[col].min() >= 0:  # decision uses training rows only
                    df_numeric[col] = np.log1p(df_numeric[col])
                    self.log_transform_cols.append(col)
                    print(f"  Applied log1p to: {col}")
        
        # Feature Engineering: Add 1-day difference and 7-day moving average for target
        # Apply after log transformation so features are in the same scale
        if target_col and target_col in df_numeric.columns:
            if self.use_panel_data and 'Station_ID' in df.columns:
                groups = df['Station_ID']
                df_numeric[f'{target_col}_diff1'] = (
                    df_numeric[target_col].groupby(groups).diff(1).fillna(0)
                )
                df_numeric[f'{target_col}_ma7'] = (
                    df_numeric[target_col].groupby(groups)
                    .transform(lambda series: series.rolling(window=7, min_periods=1).mean())
                )
            else:
                df_numeric[f'{target_col}_diff1'] = df_numeric[target_col].diff(1).fillna(0)
                df_numeric[f'{target_col}_ma7'] = (
                    df_numeric[target_col].rolling(window=7, min_periods=1).mean()
                )
            
            # Update feature columns list
            feature_cols.extend([f'{target_col}_diff1', f'{target_col}_ma7'])
            print(f"  Added engineered features: {target_col}_diff1, {target_col}_ma7")
        
        self.feature_names = feature_cols

        # Normalize features
        self.scaler.fit(df_numeric.loc[fit_mask])
        df_normalized = pd.DataFrame(
            self.scaler.transform(df_numeric),
            columns=feature_cols,
            index=df.index
        )

        print(f"Using {len(feature_cols)} features for modeling (including engineered features)")
        return df_normalized, feature_cols
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform: denormalize and reverse log transformation
        
        Args:
            data: Normalized data (n_samples, n_features)
            
        Returns:
            data_original: Original scale data
        """
        # Ensure data has correct shape
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Denormalize
        data_denorm = self.scaler.inverse_transform(data)
        
        # Reverse log transformation (only for original features, not engineered ones)
        if self.apply_log_transform and len(self.log_transform_cols) > 0:
            for col in self.log_transform_cols:
                if col in self.feature_names:
                    col_idx = self.feature_names.index(col)
                    # Only apply expm1 to non-negative values (log1p was applied to non-negative)
                    data_denorm[:, col_idx] = np.maximum(0, np.expm1(data_denorm[:, col_idx]))
        
        return data_denorm

    def create_sequences(self, data: np.ndarray, station_ids: np.ndarray = None,
                         seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Phase 1: Create time series sequences for panel data
        
        핵심: 사이트별로 시퀀스를 생성하여 temporal leakage 방지
        (Site A의 마지막 시점과 Site B의 첫 시점이 섞이지 않도록)

        Args:
            data: Normalized feature matrix (N, n_features)
            station_ids: (N,) - Station ID for each sample (Phase 1)
            seq_len: Sequence length (lookback window)

        Returns:
            X: Shape (M, seq_len, n_features) - M sequences
            y: Shape (M, 1) - Chl-a target only (Phase 2)
            station_ids_seq: (M,) - Station ID for each sequence
        """
        X, y, station_ids_seq = [], [], []
        
        if station_ids is None:
            # Single site mode (backward compatibility)
            for i in range(len(data) - seq_len):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len, self.target_idx] if self.target_idx is not None else data[i+seq_len, 0])
            return np.array(X), np.array(y).reshape(-1, 1), None
        
        # Phase 1: Panel data mode - generate sequences within each site
        current_station = station_ids[0]
        start_idx = 0
        
        for i in range(len(data)):
            # Check if we've moved to a new site
            if station_ids[i] != current_station:
                # Generate sequences for the previous site
                site_data = data[start_idx:i]
                site_station = current_station
                
                for j in range(len(site_data) - seq_len):
                    X.append(site_data[j:j+seq_len])
                    # Phase 2: Only predict Chl-a (target variable)
                    target_val = site_data[j+seq_len, self.target_idx] if self.target_idx is not None else site_data[j+seq_len, 0]
                    y.append(target_val)
                    station_ids_seq.append(site_station)
                
                # Start new site
                current_station = station_ids[i]
                start_idx = i
        
        # Handle the last site
        site_data = data[start_idx:]
        site_station = current_station
        for j in range(len(site_data) - seq_len):
            X.append(site_data[j:j+seq_len])
            target_val = site_data[j+seq_len, self.target_idx] if self.target_idx is not None else site_data[j+seq_len, 0]
            y.append(target_val)
            station_ids_seq.append(site_station)
        
        return np.array(X), np.array(y).reshape(-1, 1), np.array(station_ids_seq)

    @staticmethod
    def create_sequence_dates(dates: np.ndarray, station_ids: np.ndarray = None,
                              seq_len: int = 30) -> np.ndarray:
        """Align target dates with sequences, resetting the lookback per site.

        Panel data are stacked by station.  A single global ``dates[seq_len:]``
        slice therefore drops the lookback only once and mislabels every site
        after the first.  This helper mirrors ``create_sequences`` and drops
        ``seq_len`` dates independently inside each contiguous station block.
        """
        dates = np.asarray(dates)
        if station_ids is None:
            return dates[seq_len:]
        station_ids = np.asarray(station_ids)
        if len(dates) != len(station_ids):
            raise ValueError("dates and station_ids must have equal length")
        aligned = []
        for station_id in np.unique(station_ids):
            site_dates = dates[station_ids == station_id]
            aligned.extend(site_dates[seq_len:])
        return np.asarray(aligned)

    def split_train_test(self, df: pd.DataFrame, df_normalized: pd.DataFrame,
                        seq_len: int = 30,
                        train_period: Tuple[str, str] = ('2013-01-01', '2022-12-31'),
                        val_period: Tuple[str, str] = ('2023-01-01', '2023-12-31'),
                        test_period: Tuple[str, str] = ('2024-01-01', '2024-12-31')) -> Dict:
        """
        Split data into train, validation, and test sets using configurable periods.
        """
        train_start, train_end = pd.to_datetime(train_period[0]), pd.to_datetime(train_period[1])
        val_start, val_end = pd.to_datetime(val_period[0]), pd.to_datetime(val_period[1])
        test_start, test_end = pd.to_datetime(test_period[0]), pd.to_datetime(test_period[1])

        train_mask = (df['측정일'] >= train_start) & (df['측정일'] <= train_end)
        val_mask = (df['측정일'] >= val_start) & (df['측정일'] <= val_end)
        test_mask = (df['측정일'] >= test_start) & (df['측정일'] <= test_end)

        train_data = df_normalized[train_mask].values
        val_data = df_normalized[val_mask].values
        test_data = df_normalized[test_mask].values

        train_station_ids_seq = None
        val_station_ids_seq = None
        test_station_ids_seq = None

        # Phase 1: Get Station_IDs for panel data
        if self.use_panel_data and 'Station_ID' in df.columns:
            train_station_ids = df[train_mask]['Station_ID'].values
            val_station_ids = df[val_mask]['Station_ID'].values
            test_station_ids = df[test_mask]['Station_ID'].values
            
            # Create sequences with site-aware generation
            X_train, y_train, train_station_ids_seq = self.create_sequences(
                train_data, train_station_ids, seq_len
            )
            X_val, y_val, val_station_ids_seq = self.create_sequences(
                val_data, val_station_ids, seq_len
            )
            X_test, y_test, test_station_ids_seq = self.create_sequences(
                test_data, test_station_ids, seq_len
            )
            
            print(f"\nPhase 1: Panel Data Split Summary")
            print(f"  Training: {X_train.shape[0]} sequences ({train_period[0]} ~ {train_period[1]})")
            print(f"  Validation: {X_val.shape[0]} sequences ({val_period[0]} ~ {val_period[1]})")
            print(f"  Test: {X_test.shape[0]} sequences ({test_period[0]} ~ {test_period[1]})")
        else:
            # Single site mode
            X_train, y_train, _ = self.create_sequences(train_data, None, seq_len)
            X_val, y_val, _ = self.create_sequences(val_data, None, seq_len)
            X_test, y_test, _ = self.create_sequences(test_data, None, seq_len)
            print(f"Training set: {X_train.shape[0]} sequences")
            print(f"Validation set: {X_val.shape[0]} sequences")
            print(f"Test set: {X_test.shape[0]} sequences")

        # Align target dates with site-aware sequence generation.  Slicing the
        # station-stacked panel globally mislabels all stations after the first.
        val_dates = self.create_sequence_dates(
            df[val_mask]['측정일'].values,
            val_station_ids if self.use_panel_data and 'Station_ID' in df.columns else None,
            seq_len,
        )
        test_dates = self.create_sequence_dates(
            df[test_mask]['측정일'].values,
            test_station_ids if self.use_panel_data and 'Station_ID' in df.columns else None,
            seq_len,
        )

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'train_station_ids': train_station_ids_seq,
            'val_station_ids': val_station_ids_seq,
            'test_station_ids': test_station_ids_seq,
            'val_dates': val_dates,
            'test_dates': test_dates,
            'original_df': df,
            'df_normalized': df_normalized
        }
