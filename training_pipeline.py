"""
Training and Inference Pipeline for TF Binding Transformer
Handles model training, evaluation, and binding site prediction

This module provides:
1. Multi-task loss functions for site detection, TF classification, and energy prediction
2. Training loop with validation and checkpointing
3. Inference functions for predicting binding sites
4. Evaluation metrics and visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report
from tqdm import tqdm

# Make wandb optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from transformer_model import TFBindingTransformer, ModelConfig, create_model
from data_preprocessing import create_data_loaders, HMMDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining site detection, TF classification, and energy prediction"""
    
    def __init__(self, site_weight: float = 1.0, tf_weight: float = 1.0, energy_weight: float = 0.5):
        super().__init__()
        self.site_weight = site_weight
        self.tf_weight = tf_weight
        self.energy_weight = energy_weight
        
        self.site_loss_fn = nn.BCELoss(reduction='none')
        self.tf_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.energy_loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                labels: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model predictions dictionary
            labels: Ground truth labels dictionary  
            mask: Attention mask for valid positions
            
        Returns:
            Dictionary with individual and total losses
        """
        batch_size, seq_len = mask.shape
        
        # Site detection loss (binary classification)
        site_pred = predictions['site_probs'][:, :, 1]  # Probability of being a site
        site_target = labels['site_labels']
        site_loss = self.site_loss_fn(site_pred, site_target)
        site_loss = (site_loss * mask).sum() / mask.sum()
        
        # TF classification loss (only on positive sites)
        tf_pred = predictions['tf_logits']  # [batch, seq_len, num_tf_types]
        tf_target = labels['tf_labels']  # [batch, seq_len]
        
        # Create mask for positions with binding sites
        site_mask = (site_target > 0.5) * mask
        if site_mask.sum() > 0:
            # Flatten for cross-entropy loss
            tf_pred_flat = tf_pred.view(-1, tf_pred.size(-1))  # [batch*seq_len, num_tf_types]
            tf_target_flat = tf_target.view(-1)  # [batch*seq_len]
            site_mask_flat = site_mask.view(-1)  # [batch*seq_len]
            
            tf_loss_per_pos = self.tf_loss_fn(tf_pred_flat, tf_target_flat)
            tf_loss = (tf_loss_per_pos * site_mask_flat).sum() / site_mask_flat.sum()
        else:
            tf_loss = torch.tensor(0.0, device=site_pred.device)
        
        # Energy prediction loss (regression, only on positive sites)
        energy_pred = predictions['binding_energy'][:, :, 0]  # [batch, seq_len]
        energy_target = labels['energy_labels']  # [batch, seq_len]
        
        energy_loss_per_pos = self.energy_loss_fn(energy_pred, energy_target)
        if site_mask.sum() > 0:
            energy_loss = (energy_loss_per_pos * site_mask).sum() / site_mask.sum()
        else:
            energy_loss = torch.tensor(0.0, device=site_pred.device)
        
        # Combine losses
        total_loss = (self.site_weight * site_loss + 
                     self.tf_weight * tf_loss + 
                     self.energy_weight * energy_loss)
        
        return {
            'total_loss': total_loss,
            'site_loss': site_loss,
            'tf_loss': tf_loss,
            'energy_loss': energy_loss
        }


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model: TFBindingTransformer, config: ModelConfig, 
                 device: str = None, use_wandb: bool = False):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = MultiTaskLoss()
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project="tf-binding-transformer",
                config=config.__dict__,
                name=f"transformer_{config.num_layers}L_{config.hidden_dim}H"
            )
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'site': 0, 'tf': 0, 'energy': 0}
        num_batches = 0
        
        with tqdm(train_loader, desc="Training") as pbar:
            for batch in pbar:
                # Move batch to device
                dna_tokens = batch['dna_tokens'].to(self.device)
                expression = batch['expression'].to(self.device)
                site_labels = batch['site_labels'].to(self.device)
                tf_labels = batch['tf_labels'].to(self.device)
                energy_labels = batch['energy_labels'].to(self.device)
                
                # Forward pass
                predictions = self.model(dna_tokens, expression)
                
                # Compute loss
                labels = {
                    'site_labels': site_labels,
                    'tf_labels': tf_labels,
                    'energy_labels': energy_labels
                }
                
                losses = self.criterion(predictions, labels, predictions['attention_mask'])
                
                # Backward pass
                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update metrics
                epoch_losses['total'] += losses['total_loss'].item()
                epoch_losses['site'] += losses['site_loss'].item()
                epoch_losses['tf'] += losses['tf_loss'].item()
                epoch_losses['energy'] += losses['energy_loss'].item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'site': f"{losses['site_loss'].item():.4f}",
                    'tf': f"{losses['tf_loss'].item():.4f}"
                })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = {'total': 0, 'site': 0, 'tf': 0, 'energy': 0}
        num_batches = 0
        
        all_site_preds = []
        all_site_labels = []
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as pbar:
                for batch in pbar:
                    # Move batch to device
                    dna_tokens = batch['dna_tokens'].to(self.device)
                    expression = batch['expression'].to(self.device)
                    site_labels = batch['site_labels'].to(self.device)
                    tf_labels = batch['tf_labels'].to(self.device)
                    energy_labels = batch['energy_labels'].to(self.device)
                    
                    # Forward pass
                    predictions = self.model(dna_tokens, expression)
                    
                    # Compute loss
                    labels = {
                        'site_labels': site_labels,
                        'tf_labels': tf_labels,
                        'energy_labels': energy_labels
                    }
                    
                    losses = self.criterion(predictions, labels, predictions['attention_mask'])
                    
                    # Update metrics
                    epoch_losses['total'] += losses['total_loss'].item()
                    epoch_losses['site'] += losses['site_loss'].item()
                    epoch_losses['tf'] += losses['tf_loss'].item()
                    epoch_losses['energy'] += losses['energy_loss'].item()
                    num_batches += 1
                    
                    # Collect predictions for metrics
                    mask = predictions['attention_mask']
                    site_pred = predictions['site_probs'][:, :, 1]
                    
                    valid_preds = site_pred[mask == 1].cpu().numpy()
                    valid_labels = site_labels[mask == 1].cpu().numpy()
                    
                    all_site_preds.extend(valid_preds)
                    all_site_labels.extend(valid_labels)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Compute additional metrics
        try:
            auc = roc_auc_score(all_site_labels, all_site_preds)
            epoch_losses['auc'] = auc
        except:
            epoch_losses['auc'] = 0.0
            
        return epoch_losses
    
    def train(self, train_loader, val_loader, num_epochs: int, 
              save_dir: str = "checkpoints") -> Dict[str, List[float]]:
        """Full training loop"""
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        history = {
            'train_loss': [], 'val_loss': [], 'val_auc': [],
            'train_site_loss': [], 'train_tf_loss': [], 'train_energy_loss': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate_epoch(val_loader)
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            history['train_loss'].append(train_losses['total'])
            history['val_loss'].append(val_losses['total'])
            history['val_auc'].append(val_losses.get('auc', 0.0))
            history['train_site_loss'].append(train_losses['site'])
            history['train_tf_loss'].append(train_losses['tf'])
            history['train_energy_loss'].append(train_losses['energy'])
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_losses['total'],
                    'val_loss': val_losses['total'],
                    'val_auc': val_losses.get('auc', 0.0),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            logger.info(f"Train Loss: {train_losses['total']:.4f}, "
                       f"Val Loss: {val_losses['total']:.4f}, "
                       f"Val AUC: {val_losses.get('auc', 0.0):.4f}")
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'val_loss': best_val_loss
                }, save_path / 'best_model.pt')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': history
                }, save_path / f'checkpoint_epoch_{epoch+1}.pt')
        
        return history


class BindingSitePredictor:
    """Handles inference and binding site prediction"""
    
    def __init__(self, model: TFBindingTransformer, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict_sequence(self, sequence: str, expression_values: np.ndarray,
                        threshold: float = 0.5) -> List[Dict]:
        """Predict binding sites for a single sequence"""
        return self.model.predict_binding_sites(sequence, expression_values, threshold)
    
    def predict_batch(self, sequences: Dict[str, str], expression_data: pd.DataFrame,
                     threshold: float = 0.5) -> Dict[str, List[Dict]]:
        """Predict binding sites for multiple sequences"""
        predictions = {}
        
        for seq_id, sequence in sequences.items():
            if seq_id in expression_data.index:
                expr_values = expression_data.loc[seq_id].values
                sites = self.predict_sequence(sequence, expr_values, threshold)
                predictions[seq_id] = sites
                
        return predictions
    
    def compare_with_hmm(self, sequences: Dict[str, str], expression_data: pd.DataFrame,
                        hmm_results_file: str = None) -> pd.DataFrame:
        """Compare transformer predictions with original HMM results"""
        transformer_predictions = self.predict_batch(sequences, expression_data)
        
        # Create comparison dataframe
        comparison_data = []
        
        for seq_id, sites in transformer_predictions.items():
            for site in sites:
                comparison_data.append({
                    'sequence_id': seq_id,
                    'position': site['position'],
                    'tf_type': site['tf_type'],
                    'probability': site['probability'],
                    'binding_energy': site['binding_energy'],
                    'method': 'transformer'
                })
        
        return pd.DataFrame(comparison_data)


def load_trained_model(checkpoint_path: str, device: str = None) -> TFBindingTransformer:
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def main():
    """Main training script"""
    config = ModelConfig(
        max_seq_length=1024,
        num_layers=4,
        hidden_dim=256,
        num_heads=8,
        learning_rate=1e-4,
        batch_size=16
    )
    
    # Create model
    model = create_model(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    try:
        train_loader, val_loader = create_data_loaders("R0315/iData", config, batch_size=config.batch_size)
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # Create trainer
    trainer = ModelTrainer(model, config, use_wandb=False)
    
    # Train model
    history = trainer.train(train_loader, val_loader, num_epochs=50)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation') 
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_auc'])
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_site_loss'], label='Site Loss')
    plt.plot(history['train_tf_loss'], label='TF Loss')
    plt.plot(history['train_energy_loss'], label='Energy Loss')
    plt.title('Training Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Training completed! Model saved to checkpoints/best_model.pt")


if __name__ == "__main__":
    main()