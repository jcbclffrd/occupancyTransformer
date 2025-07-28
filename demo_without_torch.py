"""
Demo script showing data processing pipeline without PyTorch
Demonstrates loading and processing your HMM data for transformer input
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDataLoader:
    """Simplified data loader without PyTorch dependencies"""
    
    def __init__(self, data_dir: str = "R0315/iData"):
        self.data_dir = Path(data_dir)
        
    def load_sequences(self, fasta_file: str):
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
    
    def load_expression_data(self, expr_file: str):
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
    
    def load_motif_data(self, motif_file: str):
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
                        try:
                            values = [float(x) for x in line.split('\t')]
                            if len(values) == 4:  # A, C, G, T
                                current_matrix.append(values)
                        except ValueError:
                            continue
                
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


class SimpleTokenizer:
    """Simple DNA tokenizer without PyTorch"""
    
    def __init__(self):
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'PAD': 5}
    
    def tokenize(self, sequence: str, max_length: int = 1000):
        """Convert DNA sequence to tokens"""
        sequence = sequence.upper()
        tokens = [self.vocab.get(base, self.vocab['N']) for base in sequence]
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens.extend([self.vocab['PAD']] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]
            
        return np.array(tokens)


class SimplePWMScanner:
    """Simple PWM scanner for generating synthetic labels"""
    
    def __init__(self, motifs, threshold=0.8):
        self.motifs = motifs
        self.threshold = threshold
    
    def score_position(self, sequence: str, position: int, pwm: np.ndarray):
        """Calculate PWM score at position"""
        if position + len(pwm) > len(sequence):
            return 0.0
            
        score = 0.0
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        for i, base in enumerate(sequence[position:position + len(pwm)]):
            if base in base_to_idx:
                # Convert count matrix to frequencies and add pseudocount
                freq = pwm[i, base_to_idx[base]] + 0.25
                total = np.sum(pwm[i, :]) + 1.0
                score += np.log(freq / total)
                
        return score
    
    def scan_sequence(self, sequence: str, motif_name: str):
        """Scan sequence for binding sites"""
        if motif_name not in self.motifs:
            return []
            
        pwm = self.motifs[motif_name]
        sites = []
        
        # Calculate maximum possible score for normalization
        max_scores = []
        for i in range(len(pwm)):
            freqs = pwm[i, :] + 0.25
            total = np.sum(freqs)
            max_scores.append(np.log(np.max(freqs) / total))
        max_score = sum(max_scores)
        
        # Scan sequence
        for pos in range(len(sequence) - len(pwm) + 1):
            score = self.score_position(sequence, pos, pwm)
            normalized_score = score / max_score if max_score != 0 else 0
            
            if normalized_score >= self.threshold:
                sites.append({
                    'position': pos,
                    'motif': motif_name,
                    'score': normalized_score,
                    'length': len(pwm),
                    'sequence': sequence[pos:pos+len(pwm)]
                })
                
        return sites


def demonstrate_pipeline():
    """Demonstrate the complete data processing pipeline"""
    logger.info("=== TF Binding Transformer Data Pipeline Demo ===")
    
    # Check if data directory exists
    data_dir = Path("R0315/iData")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please ensure your HMM data is in the R0315/iData directory")
        return
    
    # Initialize data loader
    loader = SimpleDataLoader(str(data_dir))
    
    # Load sequences
    logger.info("\n1. Loading DNA sequences...")
    sequences = loader.load_sequences("efastanotw12.txt")
    
    if sequences:
        sample_id = list(sequences.keys())[0]
        sample_seq = sequences[sample_id]
        logger.info(f"   Sample sequence '{sample_id}': {len(sample_seq)} bp")
        logger.info(f"   First 100 bp: {sample_seq[:100]}")
        
        # Show sequence statistics
        lengths = [len(seq) for seq in sequences.values()]
        logger.info(f"   Sequence lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
    
    # Load expression data
    logger.info("\n2. Loading expression data...")
    expression_data = loader.load_expression_data("expre12.tab")
    
    if not expression_data.empty:
        logger.info(f"   Expression matrix: {expression_data.shape}")
        logger.info(f"   Sample conditions: {list(expression_data.columns[:5])}...")
        logger.info(f"   Sample values for '{expression_data.index[0]}': {expression_data.iloc[0, :5].values}")
    
    # Load motifs
    logger.info("\n3. Loading transcription factor motifs...")
    motifs = loader.load_motif_data("factordts.wtmx")
    
    if motifs:
        for name, matrix in list(motifs.items())[:3]:  # Show first 3 motifs
            logger.info(f"   Motif '{name}': {matrix.shape} matrix")
            logger.info(f"   First row (A,C,G,T): {matrix[0, :]}")
    
    # Demonstrate tokenization
    logger.info("\n4. DNA sequence tokenization...")
    tokenizer = SimpleTokenizer()
    
    if sequences:
        sample_seq = list(sequences.values())[0][:50]  # First 50 bases
        tokens = tokenizer.tokenize(sample_seq, max_length=50)
        logger.info(f"   Original sequence: {sample_seq}")
        logger.info(f"   Tokenized:         {tokens}")
        logger.info(f"   Token mapping: A=0, C=1, G=2, T=3, N=4, PAD=5")
    
    # Demonstrate PWM scanning
    logger.info("\n5. Synthetic label generation (PWM scanning)...")
    if motifs and sequences:
        scanner = SimplePWMScanner(motifs, threshold=0.7)
        
        # Scan first sequence with first motif
        first_seq_id = list(sequences.keys())[0]
        first_motif = list(motifs.keys())[0]
        
        sites = scanner.scan_sequence(sequences[first_seq_id], first_motif)
        logger.info(f"   Found {len(sites)} binding sites for motif '{first_motif}' in sequence '{first_seq_id}'")
        
        for i, site in enumerate(sites[:3]):  # Show first 3 sites
            logger.info(f"   Site {i+1}: pos={site['position']}, score={site['score']:.3f}, seq='{site['sequence']}'")
    
    # Show data compatibility
    logger.info("\n6. Transformer compatibility check...")
    
    compatible_sequences = []
    if sequences and not expression_data.empty:
        for seq_id in sequences.keys():
            if seq_id in expression_data.index:
                compatible_sequences.append(seq_id)
    
    logger.info(f"   Compatible sequences (have both DNA and expression): {len(compatible_sequences)}")
    logger.info(f"   Total sequences: {len(sequences)}")
    logger.info(f"   Expression profiles: {len(expression_data)}")
    
    if compatible_sequences:
        logger.info(f"   Sample compatible sequence: {compatible_sequences[0]}")
        
        # Show what transformer input would look like
        seq_id = compatible_sequences[0]
        dna_seq = sequences[seq_id]
        expr_values = expression_data.loc[seq_id].values
        
        logger.info(f"   DNA length: {len(dna_seq)} bp")
        logger.info(f"   Expression conditions: {len(expr_values)}")
        logger.info(f"   Expression values: {expr_values[:5]}...")
    
    # Summary
    logger.info("\n=== Summary ===")
    logger.info(f"✓ DNA sequences loaded: {len(sequences)}")
    logger.info(f"✓ Expression profiles loaded: {len(expression_data)}")
    logger.info(f"✓ TF motifs loaded: {len(motifs)}")
    logger.info(f"✓ Compatible data pairs: {len(compatible_sequences)}")
    
    if len(compatible_sequences) > 0:
        logger.info("\n🎉 Your data is fully compatible with the transformer pipeline!")
        logger.info("   Ready for model training once PyTorch is installed.")
    else:
        logger.info("\n⚠️  Data compatibility issues found.")
        logger.info("   Please check that sequence IDs match between FASTA and expression files.")
    
    return sequences, expression_data, motifs


if __name__ == "__main__":
    demonstrate_pipeline()