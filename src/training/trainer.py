
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple
import numpy as np

from src.utils.constants import get_device, cleanup_memory
from src.utils.reproducibility import make_generator, seed_worker
from src.training.callbacks import EarlyStopping
from src.models.losses import gaussian_nll_loss, adaptive_decorrelation_nll_loss, iso_nll_loss, iso_der_loss, beta_nll_loss, faithful_hr_loss

class HABTrainer:
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)
        
    def train(self, 
              X_train: torch.Tensor, y_train: torch.Tensor, 
              X_val: torch.Tensor, y_val: torch.Tensor,
              train_station_ids: torch.Tensor = None, 
              val_station_ids: torch.Tensor = None):
        """
        Train the model
        """
        print("\n[STEP 3] Training Model...")
        
        # Optimizer & Scheduler
        learning_rate = min(self.config['learning_rate'], 0.0005)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        print(f"  Using learning rate: {learning_rate}")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # DataLoaders
        use_panel_data = self.config.get('use_panel_data', True)
        
        if use_panel_data and train_station_ids is not None:
            train_dataset = TensorDataset(X_train, y_train, train_station_ids)
        else:
            train_dataset = TensorDataset(X_train, y_train)
            
        if use_panel_data and val_station_ids is not None:
            val_dataset = TensorDataset(X_val, y_val, val_station_ids)
        else:
            val_dataset = TensorDataset(X_val, y_val)
            
        use_multiprocessing = not torch.backends.mps.is_available()
        batch_size = self.config['batch_size']
        
        # Deterministic shuffling: a seeded generator makes batch order
        # reproducible across runs given a fixed config['seed'].
        seed = self.config.get('seed', 42)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12 if use_multiprocessing else 0,
            pin_memory=True if not torch.backends.mps.is_available() else False,
            persistent_workers=True if use_multiprocessing and 12 > 0 else False,
            prefetch_factor=2 if use_multiprocessing and 12 > 0 else None,
            generator=make_generator(seed),
            worker_init_fn=seed_worker
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Early Stopping
        early_stopping = EarlyStopping(
            patience=self.config.get('patience', 15), 
            verbose=True, 
            path=self.config['model_save_path']
        )
        
        # Training Parameters
        epochs = self.config['epochs']
        # User requested warm-up for first 5-10 epochs. 
        # We set it to 10 to ensure stable backbone features before penalties kick in.
        warmup_epochs = 10 
        stage2_triggered = False
        
        for epoch in range(epochs):
            # Stage logic
            use_iso_nll = self.config.get('use_iso_nll', False)
            uq_method = self.config.get('uq_method', '')
            model_type = self.config.get('model_type', '')
            
            # Default ISO parameters
            curr_lambda_indep = 0.0
            curr_lambda_sharp = 0.0
            current_lambd = 0.0 # Legacy for Phase 2
            current_alpha = 0.3
            
            # === 3-Stage Curriculum for ISO-DER ===
            is_isoder_experiment = (model_type == 'MambaISODER' or uq_method == 'ISODER')
            target_lambda_indep = self.config.get('lambd_indep', 0.1)
            target_lambda_sharp = self.config.get('lambd_sharp', 0.05)
            
            # [NEW] Configuration for curriculum
            use_curriculum = self.config.get('use_curriculum', True) if is_isoder_experiment else False
            
            # Update dynamic penalties if using curriculum
            if use_curriculum:
                # Stage 1: Warm-up (Standard DER) - Epoch 0-10
                if epoch < 10:
                    curr_lambda_indep = 0.0
                    curr_lambda_sharp = 0.0
                    if epoch == 0: print("\n[Stage 1] Warm-up: Standard DER Training")
                    
                # Stage 2: Decoupling (Independence) - Epoch 11-50
                elif epoch < 50:
                    curr_lambda_indep = target_lambda_indep
                    curr_lambda_sharp = 0.0
                    if epoch == 10: print(f"\n[Stage 2] Decoupling: Adding Independence Penalty ({curr_lambda_indep})")
                
                # Stage 3: Sharpening (Width Minimization) - Epoch 51+
                else:
                    curr_lambda_indep = target_lambda_indep
                    curr_lambda_sharp = target_lambda_sharp
                    if epoch == 50: print(f"\n[Stage 3] Sharpening: Adding Sharpness Penalty ({curr_lambda_sharp})")
            else:
                # Use target values directly (No Curriculum)
                curr_lambda_indep = target_lambda_indep if (is_isoder_experiment or use_iso_nll) else 0.0
                curr_lambda_sharp = target_lambda_sharp if (is_isoder_experiment or use_iso_nll) else 0.0
            
            # === Legacy Phase 2 Logic ===
            if not is_isoder_experiment and epoch < warmup_epochs:
                 current_lambd = 0.0
            elif not is_isoder_experiment:
                 current_lambd = 0.1
                 # For non-ISODER Phase 2, curr_lambda values might be updated here if use_iso_nll
                 if use_iso_nll:
                     curr_lambda_indep = self.config.get('lambd_indep', 0.1)
                     curr_lambda_sharp = self.config.get('lambd_sharp', 0.05)
                 
                 if not stage2_triggered:
                    if use_iso_nll:
                        print(f"\n[Transition] Phase 2 (ISO-NLL): Indep={curr_lambda_indep}, Sharp={curr_lambda_sharp}")
                    else:
                        print(f"\n[Transition] Switching to Stage 2: Adaptive-Detach (Lambda={current_lambd}, Alpha={current_alpha})")
                    print("  Reducing Learning Rate by 50%...")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    stage2_triggered = True
            
            self.model.train()
            train_loss = 0
            
            for batch_data in train_loader:
                optimizer.zero_grad()
                
                if use_panel_data and len(batch_data) == 3:
                    batch_X, batch_y, batch_ids = batch_data
                else:
                    batch_X, batch_y = batch_data
                    batch_ids = None
                    
                mean, var = self.model(batch_X, station_ids=batch_ids, mc_samples=1)
                
                if self.config.get('uq_method') in ['DER', 'DER_Best', 'DER_Amini'] or self.config.get('model_type') == 'MambaISODER':
                    # DER Logic
                    # model(mc_samples=1) calls forward -> returns (gamma, total_var)
                    # BUT for DER loss we need 4 params.
                    # We need to call get_evidential_params or modify forward logic for training
                    
                    if hasattr(self.model, 'get_evidential_params'):
                        gamma, nu, alpha, beta = self.model.get_evidential_params(batch_X, station_ids=batch_ids)
                        
                        if self.config.get('model_type') == 'MambaISODER':
                             # ISO-DER Loss
                             total_loss, logs = iso_der_loss(
                                 batch_y, gamma, nu, alpha, beta,
                                 lambd_indep=curr_lambda_indep,
                                 lambd_sharp=curr_lambda_sharp
                             )
                             loss = total_loss
                        else:
                             # Standard DER derived via iso_der_loss with 0 penalties (or custom)
                             # [NEW] Amini et al. / Best Configuration:
                             # Warm-up: KL penalty kicks in after warm-up for Amini, or immediate for Best
                             kl_coeff = 0.0
                             uq_method = self.config.get('uq_method', '')
                             
                             if uq_method == 'DER_Best':
                                 kl_coeff = 0.01 # Always apply for Best reproduction
                             elif uq_method == 'DER_Amini' and epoch >= warmup_epochs:
                                 kl_coeff = 0.01 
                                 
                             total_loss, logs = iso_der_loss(
                                 batch_y, gamma, nu, alpha, beta,
                                 lambd_indep=0.0, lambd_sharp=0.0,
                                 coeff=kl_coeff # Inject KL coeff
                             )
                             loss = total_loss
                    else:
                        raise ValueError("DER model must implement get_evidential_params")
                        
                elif uq_method == 'BetaNLL':
                    # Ablation baseline: beta-NLL (Seitzer et al., ICLR 2022)
                    loss = beta_nll_loss(batch_y, mean, var, beta=self.config.get('beta_nll', 0.5))
                elif uq_method == 'Faithful':
                    # Ablation baseline: Faithful Heteroscedastic Regression (Stirn et al., AISTATS 2023)
                    loss = faithful_hr_loss(batch_y, mean, var)
                elif use_iso_nll:
                    loss, logs = iso_nll_loss(
                        batch_y, mean, var,
                        lambd_indep=curr_lambda_indep,
                        lambd_sharp=curr_lambda_sharp,
                        alpha=current_alpha
                    )
                else:
                    loss = adaptive_decorrelation_nll_loss(
                        batch_y, mean, var, 
                        alpha=current_alpha, lambd=current_lambd
                    )
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    if use_panel_data and len(batch_data) == 3:
                        batch_X, batch_y, batch_ids = batch_data
                    else:
                        batch_X, batch_y = batch_data
                        batch_ids = None
                        
                    if self.config.get('uq_method') in ['DER', 'DER_Best', 'DER_Amini'] or self.config.get('model_type') == 'MambaISODER':
                        # DER Validation Logic
                        if hasattr(self.model, 'get_evidential_params'):
                            gamma, nu, alpha, beta = self.model.get_evidential_params(batch_X, station_ids=batch_ids)
                            
                            if self.config.get('model_type') == 'MambaISODER':
                                loss, _ = iso_der_loss(
                                    batch_y, gamma, nu, alpha, beta,
                                    lambd_indep=curr_lambda_indep,
                                    lambd_sharp=curr_lambda_sharp
                                )
                            else:
                                kl_coeff = 0.0
                                uq_method = self.config.get('uq_method', '')
                                if uq_method == 'DER_Best':
                                    kl_coeff = 0.01
                                elif uq_method == 'DER_Amini' and epoch >= warmup_epochs:
                                    kl_coeff = 0.01
                                
                                loss, _ = iso_der_loss(
                                    batch_y, gamma, nu, alpha, beta,
                                    lambd_indep=0.0, lambd_sharp=0.0,
                                    coeff=kl_coeff
                                )
                        else:
                            raise ValueError("DER model must implement get_evidential_params")
                    else:
                        # Standard Gaussian NLL Validation
                        mean, var = self.model(batch_X, station_ids=batch_ids, mc_samples=1)
                        loss = gaussian_nll_loss(batch_y, mean, var)
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            scheduler.step(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                
            early_stopping(avg_val_loss, self.model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Load best model
        print(f"Loading best model from {self.config['model_save_path']}...")
        self.model.load_state_dict(torch.load(self.config['model_save_path']))
        
        cleanup_memory()
        print(f"\nBest model saved to {self.config['model_save_path']}")
        
        return self.model
