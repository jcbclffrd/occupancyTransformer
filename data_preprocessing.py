"""
Data Preprocessing Pipeline for Transformer Model
Converts HMM framework data formats to transformer-compatible tensors

This module handles:
1. Loading DNA sequences from FASTA files
2. Loading expression data from tab-delimited files  
3. Loading PWM motifs and generating synthetic training labels
4. Creating batched datasets for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path
from transformer_model import ModelConfig, DNATokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HMMDataLoader:
    """Loads data from the original HMM framework format"""
    
    def __init__(self, data_dir: str = "R0315/iData"):
        self.data_dir = Path(data_dir)
        
    def load_sequences(self, fasta_file: str) -> Dict[str, str]:
        """Load DNA sequences from FASTA file"""
        sequences = {}
        current_id = None
        current_seq = []
        
        fasta_path = self.data_dir / fasta_file
        
        try:
            with open(fasta_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        # Save previous sequence
                        if current_id is not None:
                            sequences[current_id] = ''.join(current_seq)
                        
                        # Start new sequence
                        current_id = line[1:]  # Remove '>'
                        current_seq = []
                    else:
                        current_seq.append(line.upper())
                
                # Save last sequence
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                    
        except FileNotFoundError:
            logger.error(f"FASTA file not found: {fasta_path}")
            return {}
            
        logger.info(f"Loaded {len(sequences)} sequences from {fasta_file}")
        return sequences
    
    def load_expression_data(self, expr_file: str) -> pd.DataFrame:
        """Load expression data from tab-delimited file"""
        expr_path = self.data_dir / expr_file
        
        try:
            # Load the expression data - first column is sequence names
            df = pd.read_csv(expr_path, sep='\t', index_col=0)
            logger.info(f"Loaded expression data: {df.shape[0]} sequences, {df.shape[1]} conditions")
            return df
        except FileNotFoundError:
            logger.error(f"Expression file not found: {expr_path}")
            return pd.DataFrame()
    
    def load_motif_data(self, motif_file: str) -> Dict[str, np.ndarray]:
        """Load PWM motifs from the motif file format"""
        motifs = {}
        motif_path = self.data_dir / motif_file
        
        try:
            with open(motif_path, 'r') as f:
                current_motif = None
                current_matrix = []
                
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        # Save previous motif
                        if current_motif is not None and current_matrix:
                            motifs[current_motif] = np.array(current_matrix)
                        
                        # Extract motif name and dimensions
                        parts = line[1:].split('\t')
                        current_motif = parts[0]
                        current_matrix = []
                        
                    elif line.startswith('<'):
                        # End of current motif
                        if current_motif is not None and current_matrix:
                            motifs[current_motif] = np.array(current_matrix)
                        current_motif = None
                        current_matrix = []
                        
                    elif line and current_motif is not None:
                        # Parse matrix row
                        values = [float(x) for x in line.split('\t')]
                        if len(values) == 4:  # A, C, G, T
                            current_matrix.append(values)
                
                # Save last motif
                if current_motif is not None and current_matrix:
                    motifs[current_motif] = np.array(current_matrix)
                    
        except FileNotFoundError:
            logger.error(f"Motif file not found: {motif_path}")
            return {}
            
        logger.info(f"Loaded {len(motifs)} motifs")
        for name, matrix in motifs.items():
            logger.info(f"  {name}: {matrix.shape[0]} positions")
            
        return motifs


class SyntheticLabelGenerator:
    """Generates training labels by scanning sequences with PWMs"""
    
    def __init__(self, motifs: Dict[str, np.ndarray], threshold: float = 0.8):
        self.motifs = motifs
        self.threshold = threshold
        self.motif_names = list(motifs.keys())
    
    def score_position(self, sequence: str, position: int, pwm: np.ndarray) -> float:
        """Calculate PWM score at a specific position"""
        if position + len(pwm) > len(sequence):
            return 0.0
            
        score = 0.0
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        for i, base in enumerate(sequence[position:position + len(pwm)]):
            if base in base_to_idx:
                score += np.log(pwm[i, base_to_idx[base]] + 1e-8)  # Add pseudocount
            
        return score
    
    def scan_sequence(self, sequence: str, motif_name: str) -> List[Dict]:
        """Scan sequence for binding sites of a specific motif"""
        if motif_name not in self.motifs:
            return []
            
        pwm = self.motifs[motif_name]
        sites = []
        
        # Calculate maximum possible score for normalization
        max_score = sum(np.log(np.max(pwm[i, :]) + 1e-8) for i in range(len(pwm)))
        
        # Scan both strands
        for strand in ['+', '-']:
            scan_seq = sequence if strand == '+' else self._reverse_complement(sequence)
            
            for pos in range(len(scan_seq) - len(pwm) + 1):
                score = self.score_position(scan_seq, pos, pwm)
                normalized_score = score / max_score if max_score != 0 else 0
                
                if normalized_score >= self.threshold:
                    actual_pos = pos if strand == '+' else len(sequence) - pos - len(pwm)
                    sites.append({
                        'position': actual_pos,
                        'strand': strand,
                        'motif': motif_name,
                        'score': normalized_score,
                        'length': len(pwm)
                    })
                    
        return sites
    
    def _reverse_complement(self, sequence: str) -> str:
        """Get reverse complement of DNA sequence"""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in reversed(sequence))
    
    def generate_labels(self, sequences: Dict[str, str]) -> Dict[str, List[Dict]]:
        """Generate binding site labels for all sequences"""
        all_labels = {}
        
        for seq_id, sequence in sequences.items():
            seq_sites = []
            for motif_name in self.motif_names:
                sites = self.scan_sequence(sequence, motif_name)
                seq_sites.extend(sites)
            
            # Sort sites by position
            seq_sites.sort(key=lambda x: x['position'])
            all_labels[seq_id] = seq_sites
            
        total_sites = sum(len(sites) for sites in all_labels.values())
        logger.info(f"Generated {total_sites} binding site labels across {len(sequences)} sequences")
        
        return all_labels


class TFBindingDataset(Dataset):
    """PyTorch dataset for transformer training"""
    
    def __init__(self, sequences: Dict[str, str], expression_data: pd.DataFrame, 
                 binding_labels: Dict[str, List[Dict]], config: ModelConfig):
        self.sequences = sequences
        self.expression_data = expression_data
        self.binding_labels = binding_labels
        self.config = config
        self.tokenizer = DNATokenizer()
        
        # Filter sequences that have both expression data and binding labels
        self.seq_ids = [
            seq_id for seq_id in sequences.keys() 
            if seq_id in expression_data.index and seq_id in binding_labels
        ]
        
        logger.info(f"Dataset created with {len(self.seq_ids)} sequences")
    
    def __len__(self) -> int:
        return len(self.seq_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_id = self.seq_ids[idx]
        sequence = self.sequences[seq_id]
        
        # Tokenize DNA sequence
        dna_tokens = self.tokenizer.encode(sequence, self.config.max_seq_length)
        
        # Get expression data
        expr_values = self.expression_data.loc[seq_id].values.astype(np.float32)
        
        # Pad expression data if needed
        if len(expr_values) < self.config.max_conditions:
            expr_padded = np.zeros(self.config.max_conditions, dtype=np.float32)
            expr_padded[:len(expr_values)] = expr_values
            expr_values = expr_padded
        else:
            expr_values = expr_values[:self.config.max_conditions]
            
        expr_tensor = torch.tensor(expr_values, dtype=torch.float32)
        
        # Create labels
        labels = self._create_labels(sequence, self.binding_labels[seq_id])
        
        return {
            'dna_tokens': dna_tokens,
            'expression': expr_tensor,
            'site_labels': labels['site_labels'],
            'tf_labels': labels['tf_labels'], 
            'energy_labels': labels['energy_labels'],
            'seq_id': seq_id
        }
    
    def _create_labels(self, sequence: str, sites: List[Dict]) -> Dict[str, torch.Tensor]:
        """Create label tensors from binding site annotations"""
        # Always use max_seq_length for consistent tensor sizes
        seq_length = self.config.max_seq_length
        
        # Initialize labels
        site_labels = torch.zeros(seq_length, dtype=torch.float32)  # Binary site/no-site
        tf_labels = torch.zeros(seq_length, dtype=torch.long)  # TF type indices
        energy_labels = torch.zeros(seq_length, dtype=torch.float32)  # Binding energies
        
        # Map motif names to indices
        motif_to_idx = {name: idx for idx, name in enumerate(set(site['motif'] for site in sites))}
        
        for site in sites:
            pos = site['position']
            if pos < seq_length:
                site_labels[pos] = 1.0
                tf_labels[pos] = motif_to_idx.get(site['motif'], 0)
                energy_labels[pos] = float(site['score'])
                
                # Mark positions within binding site
                site_length = site.get('length', 10)  # Default length
                for offset in range(1, min(site_length, seq_length - pos)):
                    if pos + offset < seq_length:
                        site_labels[pos + offset] = 0.5  # Mark as part of site
        
        return {
            'site_labels': site_labels,
            'tf_labels': tf_labels,
            'energy_labels': energy_labels
        }


def create_data_loaders(data_dir: str, config: ModelConfig, 
                       batch_size: int = None, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    if batch_size is None:
        batch_size = config.batch_size
    
    # Load data
    loader = HMMDataLoader(data_dir)
    sequences = loader.load_sequences("efastanotw12.txt")
    expression_data = loader.load_expression_data("expre12.tab")
    motifs = loader.load_motif_data("factordts.wtmx")
    
    if not sequences or expression_data.empty or not motifs:
        raise ValueError("Failed to load required data files")
    
    # Generate synthetic labels
    label_generator = SyntheticLabelGenerator(motifs)
    binding_labels = label_generator.generate_labels(sequences)
    
    # Create dataset
    dataset = TFBindingDataset(sequences, expression_data, binding_labels, config)
    
    # Split into train/validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )
    
    logger.info(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading
    config = ModelConfig()
    
    try:
        train_loader, val_loader = create_data_loaders("R0315/iData", config, batch_size=4)
        
        # Test a batch
        for batch in train_loader:
            print(f"DNA tokens shape: {batch['dna_tokens'].shape}")
            print(f"Expression shape: {batch['expression'].shape}")
            print(f"Site labels shape: {batch['site_labels'].shape}")
            print(f"TF labels shape: {batch['tf_labels'].shape}")
            print(f"Energy labels shape: {batch['energy_labels'].shape}")
            break
            
    except Exception as e:
        logger.error(f"Error in data loading: {e}")