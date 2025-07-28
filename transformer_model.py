"""
Transformer Model for Transcription Factor Binding Site Annotation
Transforms HMM-based approach to attention-based deep learning model

This model takes DNA sequences and expression data as input and predicts
transcription factor binding sites using multi-modal transformer attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the transformer model"""
    # DNA sequence parameters
    max_seq_length: int = 2048
    dna_vocab_size: int = 6  # A, C, G, T, N, pad
    dna_embed_dim: int = 128
    
    # Expression data parameters
    max_conditions: int = 40  # Based on your expre12.tab
    expr_embed_dim: int = 64
    
    # Transformer parameters
    num_heads: int = 8
    num_layers: int = 6
    hidden_dim: int = 512
    dropout: float = 0.1
    
    # Output parameters
    num_tf_types: int = 10  # Number of transcription factor types
    max_binding_sites: int = 50  # Maximum sites per sequence
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32


class DNATokenizer:
    """Tokenizer for DNA sequences"""
    
    def __init__(self):
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'PAD': 5}
        self.vocab_size = len(self.vocab)
    
    def encode(self, sequence: str, max_length: int) -> torch.Tensor:
        """Convert DNA sequence to token indices"""
        sequence = sequence.upper()
        tokens = [self.vocab.get(base, self.vocab['N']) for base in sequence]
        
        # Pad or truncate to max_length
        if len(tokens) < max_length:
            tokens.extend([self.vocab['PAD']] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]
            
        return torch.tensor(tokens, dtype=torch.long)
    
    def create_position_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create mask for padded positions"""
        return (tokens != self.vocab['PAD']).float()


class MultiModalEmbedding(nn.Module):
    """Embedding layer for DNA sequences and expression data"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # DNA sequence embedding
        self.dna_embedding = nn.Embedding(config.dna_vocab_size, config.dna_embed_dim)
        self.dna_pos_encoding = nn.Parameter(
            torch.randn(config.max_seq_length, config.dna_embed_dim)
        )
        
        # Expression data embedding
        self.expr_projection = nn.Linear(config.max_conditions, config.expr_embed_dim)
        
        # Project to common dimension
        self.dna_proj = nn.Linear(config.dna_embed_dim, config.hidden_dim)
        self.expr_proj = nn.Linear(config.expr_embed_dim, config.hidden_dim)
        
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, dna_tokens: torch.Tensor, expr_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dna_tokens: [batch_size, seq_length] - DNA sequence tokens
            expr_data: [batch_size, num_conditions] - Expression values
        
        Returns:
            [batch_size, seq_length, hidden_dim] - Embedded sequences
        """
        batch_size, seq_length = dna_tokens.shape
        
        # DNA sequence embedding with positional encoding
        dna_embed = self.dna_embedding(dna_tokens)  # [batch, seq_len, dna_embed_dim]
        dna_embed = dna_embed + self.dna_pos_encoding[:seq_length].unsqueeze(0)
        dna_embed = self.dna_proj(dna_embed)  # [batch, seq_len, hidden_dim]
        
        # Expression data embedding (broadcast to sequence length)
        expr_embed = self.expr_projection(expr_data)  # [batch, expr_embed_dim]
        expr_embed = self.expr_proj(expr_embed)  # [batch, hidden_dim]
        expr_embed = expr_embed.unsqueeze(1).expand(-1, seq_length, -1)  # [batch, seq_len, hidden_dim]
        
        # Combine DNA and expression information
        combined = dna_embed + expr_embed
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        
        return combined


class TransformerLayer(nn.Module):
    """Single transformer layer with multi-head attention"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, hidden_dim]
            mask: [batch_size, seq_length] - attention mask
        
        Returns:
            [batch_size, seq_length, hidden_dim]
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attention(x, x, x, key_padding_mask=(1-mask).bool() if mask is not None else None)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class BindingSiteDecoder(nn.Module):
    """Decoder for predicting binding sites"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Binding site detection head
        self.site_classifier = nn.Linear(config.hidden_dim, 2)  # Binary: site or no site
        
        # Transcription factor type head
        self.tf_classifier = nn.Linear(config.hidden_dim, config.num_tf_types)
        
        # Binding strength/energy head
        self.energy_regressor = nn.Linear(config.hidden_dim, 1)
        
        # Site boundaries head (start and end positions)
        self.boundary_regressor = nn.Linear(config.hidden_dim, 2)
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_length, hidden_dim]
        
        Returns:
            Dictionary with prediction heads
        """
        return {
            'site_probs': torch.sigmoid(self.site_classifier(hidden_states)),
            'tf_logits': self.tf_classifier(hidden_states),
            'binding_energy': self.energy_regressor(hidden_states),
            'boundaries': torch.sigmoid(self.boundary_regressor(hidden_states))
        }


class TFBindingTransformer(nn.Module):
    """Main transformer model for TF binding site prediction"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embedding = MultiModalEmbedding(config)
        
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        self.decoder = BindingSiteDecoder(config)
        
        self.tokenizer = DNATokenizer()
    
    def forward(self, dna_sequences: torch.Tensor, expr_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            dna_sequences: [batch_size, seq_length] - DNA tokens
            expr_data: [batch_size, num_conditions] - Expression values
        
        Returns:
            Dictionary with all prediction outputs
        """
        # Create attention mask for padded positions
        mask = self.tokenizer.create_position_mask(dna_sequences)
        
        # Embed inputs
        hidden_states = self.embedding(dna_sequences, expr_data)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, mask)
        
        # Decode to binding site predictions
        outputs = self.decoder(hidden_states)
        
        # Add mask to outputs for loss computation
        outputs['attention_mask'] = mask
        
        return outputs
    
    def predict_binding_sites(self, dna_sequence: str, expr_values: np.ndarray, 
                            threshold: float = 0.5) -> List[Dict]:
        """
        Predict binding sites for a single sequence
        
        Args:
            dna_sequence: DNA sequence string
            expr_values: Expression values array [num_conditions]
            threshold: Probability threshold for site detection
        
        Returns:
            List of binding site predictions
        """
        self.eval()
        
        with torch.no_grad():
            # Tokenize DNA sequence
            dna_tokens = self.tokenizer.encode(dna_sequence, self.config.max_seq_length).unsqueeze(0)
            
            # Prepare expression data
            expr_tensor = torch.tensor(expr_values, dtype=torch.float32).unsqueeze(0)
            if expr_tensor.shape[1] < self.config.max_conditions:
                # Pad expression data if needed
                padding = torch.zeros(1, self.config.max_conditions - expr_tensor.shape[1])
                expr_tensor = torch.cat([expr_tensor, padding], dim=1)
            
            # Forward pass
            outputs = self.forward(dna_tokens, expr_tensor)
            
            # Extract predictions
            site_probs = outputs['site_probs'][0, :, 1]  # [seq_length] - probability of being a site
            tf_probs = torch.softmax(outputs['tf_logits'][0], dim=-1)  # [seq_length, num_tf_types]
            energies = outputs['binding_energy'][0, :, 0]  # [seq_length]
            boundaries = outputs['boundaries'][0]  # [seq_length, 2]
            mask = outputs['attention_mask'][0]  # [seq_length]
            
            # Find binding sites above threshold
            binding_sites = []
            for pos in range(len(dna_sequence)):
                if pos >= len(site_probs) or mask[pos] == 0:
                    break
                    
                if site_probs[pos] > threshold:
                    tf_type = torch.argmax(tf_probs[pos]).item()
                    binding_sites.append({
                        'position': pos,
                        'tf_type': tf_type,
                        'probability': float(site_probs[pos]),
                        'binding_energy': float(energies[pos]),
                        'start_boundary': float(boundaries[pos, 0]),
                        'end_boundary': float(boundaries[pos, 1]),
                        'sequence_window': dna_sequence[max(0, pos-5):min(len(dna_sequence), pos+6)]
                    })
            
            return binding_sites


def create_model(config: ModelConfig) -> TFBindingTransformer:
    """Factory function to create the transformer model"""
    model = TFBindingTransformer(config)
    
    # Initialize parameters
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    
    return model


if __name__ == "__main__":
    # Example usage
    config = ModelConfig()
    model = create_model(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"DNA vocab size: {config.dna_vocab_size}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"Hidden dimension: {config.hidden_dim}")