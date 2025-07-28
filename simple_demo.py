"""
Simple demo using only built-in Python libraries
Demonstrates data loading and processing for transformer pipeline
"""

import os
from pathlib import Path
import math


def load_sequences(fasta_file):
    """Load DNA sequences from FASTA file"""
    sequences = {}
    current_id = None
    current_seq = []
    
    try:
        with open(fasta_file, 'r') as f:
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
        print(f"FASTA file not found: {fasta_file}")
        return {}
        
    print(f"Loaded {len(sequences)} sequences from {os.path.basename(fasta_file)}")
    return sequences


def load_expression_data(expr_file):
    """Load expression data from tab-delimited file"""
    expression = {}
    
    try:
        with open(expr_file, 'r') as f:
            lines = f.readlines()
            
            if len(lines) < 2:
                return {}
            
            # Parse header
            header = lines[0].strip().split('\t')
            conditions = header[1:]  # Skip first column (sequence names)
            
            # Parse data rows
            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    seq_id = parts[0]
                    values = []
                    for val in parts[1:]:
                        try:
                            values.append(float(val))
                        except ValueError:
                            values.append(0.0)
                    expression[seq_id] = values
                    
    except FileNotFoundError:
        print(f"Expression file not found: {expr_file}")
        return {}
        
    print(f"Loaded expression data: {len(expression)} sequences, {len(conditions) if 'conditions' in locals() else 0} conditions")
    return expression


def load_motif_data(motif_file):
    """Load PWM motifs"""
    motifs = {}
    
    try:
        with open(motif_file, 'r') as f:
            current_motif = None
            current_matrix = []
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous motif
                    if current_motif is not None and current_matrix:
                        motifs[current_motif] = current_matrix
                    
                    # Extract motif name
                    parts = line[1:].split('\t')
                    current_motif = parts[0]
                    current_matrix = []
                    
                elif line.startswith('<'):
                    # End of current motif
                    if current_motif is not None and current_matrix:
                        motifs[current_motif] = current_matrix
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
                motifs[current_motif] = current_matrix
                
    except FileNotFoundError:
        print(f"Motif file not found: {motif_file}")
        return {}
        
    print(f"Loaded {len(motifs)} motifs")
    for name, matrix in motifs.items():
        print(f"  {name}: {len(matrix)} positions")
        
    return motifs


def tokenize_dna(sequence, max_length=1000):
    """Convert DNA sequence to tokens"""
    vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'PAD': 5}
    
    sequence = sequence.upper()
    tokens = [vocab.get(base, vocab['N']) for base in sequence]
    
    # Pad or truncate
    if len(tokens) < max_length:
        tokens.extend([vocab['PAD']] * (max_length - len(tokens)))
    else:
        tokens = tokens[:max_length]
        
    return tokens


def score_pwm_position(sequence, position, pwm):
    """Calculate PWM score at position"""
    if position + len(pwm) > len(sequence):
        return 0.0
        
    score = 0.0
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    for i, base in enumerate(sequence[position:position + len(pwm)]):
        if base in base_to_idx:
            # Convert count to frequency with pseudocount
            row = pwm[i]
            freq = row[base_to_idx[base]] + 0.25
            total = sum(row) + 1.0
            score += math.log(freq / total)
            
    return score


def scan_sequence_for_sites(sequence, motif_name, pwm, threshold=0.7):
    """Scan sequence for binding sites"""
    sites = []
    
    # Calculate maximum possible score
    max_scores = []
    for row in pwm:
        freqs = [x + 0.25 for x in row]
        total = sum(freqs)
        max_scores.append(math.log(max(freqs) / total))
    max_score = sum(max_scores)
    
    # Scan sequence
    for pos in range(len(sequence) - len(pwm) + 1):
        score = score_pwm_position(sequence, pos, pwm)
        normalized_score = score / max_score if max_score != 0 else 0
        
        if normalized_score >= threshold:
            sites.append({
                'position': pos,
                'motif': motif_name,
                'score': normalized_score,
                'length': len(pwm),
                'sequence': sequence[pos:pos+len(pwm)]
            })
            
    return sites


def main():
    """Main demo function"""
    print("=== TF Binding Transformer Data Pipeline Demo ===")
    print("(Using built-in Python libraries only)")
    
    # Check data directory
    data_dir = Path("R0315/iData")
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please ensure your HMM data is in the R0315/iData directory")
        return
    
    print(f"Data directory found: {data_dir}")
    
    # List available files
    print(f"\nAvailable files in {data_dir}:")
    for file in data_dir.iterdir():
        if file.is_file():
            print(f"  {file.name} ({file.stat().st_size} bytes)")
    
    # Load sequences
    print("\n1. Loading DNA sequences...")
    fasta_file = data_dir / "efastanotw12.txt"
    sequences = load_sequences(str(fasta_file))
    
    if sequences:
        sample_id = list(sequences.keys())[0]
        sample_seq = sequences[sample_id]
        print(f"   Sample sequence '{sample_id}': {len(sample_seq)} bp")
        print(f"   First 100 bp: {sample_seq[:100]}")
        
        # Show sequence statistics
        lengths = [len(seq) for seq in sequences.values()]
        print(f"   Sequence lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    
    # Load expression data
    print("\n2. Loading expression data...")
    expr_file = data_dir / "expre12.tab"
    expression_data = load_expression_data(str(expr_file))
    
    if expression_data:
        sample_id = list(expression_data.keys())[0]
        sample_values = expression_data[sample_id]
        print(f"   Expression matrix: {len(expression_data)} sequences, {len(sample_values)} conditions")
        print(f"   Sample values for '{sample_id}': {sample_values[:5]}...")
    
    # Load motifs
    print("\n3. Loading transcription factor motifs...")
    motif_file = data_dir / "factordts.wtmx"
    motifs = load_motif_data(str(motif_file))
    
    if motifs:
        for i, (name, matrix) in enumerate(list(motifs.items())[:3]):  # Show first 3
            print(f"   Motif '{name}': {len(matrix)} x 4 matrix")
            print(f"   First row (A,C,G,T): {matrix[0]}")
    
    # Demonstrate tokenization
    print("\n4. DNA sequence tokenization...")
    if sequences:
        sample_seq = list(sequences.values())[0][:50]  # First 50 bases
        tokens = tokenize_dna(sample_seq, max_length=50)
        print(f"   Original sequence: {sample_seq}")
        print(f"   Tokenized:         {tokens}")
        print(f"   Token mapping: A=0, C=1, G=2, T=3, N=4, PAD=5")
    
    # Demonstrate PWM scanning
    print("\n5. Synthetic label generation (PWM scanning)...")
    if motifs and sequences:
        first_seq_id = list(sequences.keys())[0]
        first_motif_name = list(motifs.keys())[0]
        first_motif_pwm = motifs[first_motif_name]
        
        sites = scan_sequence_for_sites(
            sequences[first_seq_id], 
            first_motif_name, 
            first_motif_pwm,
            threshold=0.7
        )
        print(f"   Found {len(sites)} binding sites for motif '{first_motif_name}' in sequence '{first_seq_id}'")
        
        for i, site in enumerate(sites[:3]):  # Show first 3 sites
            print(f"   Site {i+1}: pos={site['position']}, score={site['score']:.3f}, seq='{site['sequence']}'")
    
    # Check compatibility
    print("\n6. Transformer compatibility check...")
    compatible_sequences = []
    if sequences and expression_data:
        for seq_id in sequences.keys():
            if seq_id in expression_data:
                compatible_sequences.append(seq_id)
    
    print(f"   Compatible sequences (have both DNA and expression): {len(compatible_sequences)}")
    print(f"   Total sequences: {len(sequences)}")
    print(f"   Expression profiles: {len(expression_data)}")
    
    if compatible_sequences:
        seq_id = compatible_sequences[0]
        dna_seq = sequences[seq_id]
        expr_values = expression_data[seq_id]
        
        print(f"   Sample compatible sequence: {seq_id}")
        print(f"   DNA length: {len(dna_seq)} bp")
        print(f"   Expression conditions: {len(expr_values)}")
        print(f"   Expression values: {expr_values[:5]}...")
    
    # Summary
    print("\n=== Summary ===")
    print(f"[+] DNA sequences loaded: {len(sequences)}")
    print(f"[+] Expression profiles loaded: {len(expression_data)}")
    print(f"[+] TF motifs loaded: {len(motifs)}")
    print(f"[+] Compatible data pairs: {len(compatible_sequences)}")
    
    if len(compatible_sequences) > 0:
        print("\n*** Your data is fully compatible with the transformer pipeline! ***")
        print("   Ready for model training once PyTorch is installed.")
        
        # Show what the transformer would receive as input
        print("\n   Transformer Input Preview:")
        seq_id = compatible_sequences[0]
        dna_tokens = tokenize_dna(sequences[seq_id], max_length=100)
        expr_values = expression_data[seq_id]
        
        print(f"   DNA tokens (first 20): {dna_tokens[:20]}")
        print(f"   Expression vector: {len(expr_values)} dimensions")
        print(f"   Ready for multi-modal attention!")
        
    else:
        print("\n[!] Data compatibility issues found.")
        print("   Please check that sequence IDs match between FASTA and expression files.")


if __name__ == "__main__":
    main()