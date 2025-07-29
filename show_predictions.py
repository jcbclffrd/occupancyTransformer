"""
Demonstrate what the trained transformer model outputs
Shows the difference between HMM Viterbi sequences and transformer predictions
"""

import torch
from transformer_model import ModelConfig, create_model
from training_pipeline import load_trained_model, BindingSitePredictor
from data_preprocessing import HMMDataLoader
import numpy as np


def show_transformer_output():
    """Show what the transformer actually predicts"""
    
    print("="*70)
    print("  TRANSFORMER MODEL OUTPUT DEMONSTRATION")
    print("="*70)
    
    # Load the trained model
    try:
        model = load_trained_model('checkpoints/best_model.pt')
        predictor = BindingSitePredictor(model)
        print("[OK] Loaded trained transformer model")
    except:
        print("[ERROR] No trained model found. Please train first with:")
        print("   python main.py --mode train --epochs 10")
        return
    
    # Load data (same files as your HMM)
    loader = HMMDataLoader('R0315/iData')
    sequences = loader.load_sequences("efastanotw12.txt")
    expression_data = loader.load_expression_data("expre12.tab")
    
    if not sequences or expression_data.empty:
        print("[ERROR] Could not load data files")
        return
    
    print(f"[OK] Loaded {len(sequences)} sequences and {len(expression_data)} expression profiles")
    
    # Show predictions for first sequence
    seq_id = list(sequences.keys())[0]
    sequence = sequences[seq_id]
    expr_values = expression_data.loc[seq_id].values
    
    print(f"\n📊 PREDICTIONS FOR SEQUENCE: {seq_id}")
    print(f"   Length: {len(sequence)} bp")
    print(f"   Expression conditions: {len(expr_values)}")
    print(f"   First 100 bp: {sequence[:100]}")
    
    # Generate predictions
    sites = predictor.predict_sequence(sequence, expr_values, threshold=0.5)
    
    print(f"\n🎯 TRANSFORMER PREDICTIONS:")
    print(f"   Found {len(sites)} binding sites (threshold=0.5)")
    
    if sites:
        print("\n   Top 10 binding sites:")
        print("   Pos | TF Type | Probability | Energy | Sequence Window")
        print("   ----|---------|-------------|--------|----------------")
        
        for i, site in enumerate(sites[:10]):
            print(f"   {site['position']:3d} | {site['tf_type']:7d} | "
                  f"{site['probability']:10.3f} | {site['binding_energy']:6.2f} | "
                  f"{site['sequence_window']}")
    
    # Show what this means vs HMM
    print(f"\n🔄 COMPARISON: HMM vs TRANSFORMER")
    print("="*50)
    
    print("HMM (seq2exp) Output:")
    print("• Discrete sequence: [site, no-site, site, no-site, ...]")
    print("• Viterbi path: Most likely hidden state sequence")
    print("• Binary decisions: Each position is either a site or not")
    print("• Example: SSSNNNNSSSNNNSS (S=site, N=no-site)")
    
    print("\nTransformer Output:")
    print("• Probabilistic: Each position gets a probability [0-1]")
    print("• Multi-task: Site detection + TF type + binding energy")
    print("• Continuous values: 0.0 (no site) to 1.0 (strong site)")
    print("• Example probabilities: [0.1, 0.2, 0.8, 0.9, 0.1, 0.3, ...]")
    
    # Show the raw probability output
    print(f"\n📈 RAW PROBABILITY OUTPUT (first 50 positions):")
    
    # Get model in eval mode and generate raw probabilities
    model.eval()
    with torch.no_grad():
        # Tokenize sequence
        from transformer_model import DNATokenizer
        tokenizer = DNATokenizer()
        dna_tokens = tokenizer.encode(sequence, model.config.max_seq_length).unsqueeze(0)
        
        # Prepare expression data
        expr_tensor = torch.tensor(expr_values, dtype=torch.float32).unsqueeze(0)
        if expr_tensor.shape[1] < model.config.max_conditions:
            padding = torch.zeros(1, model.config.max_conditions - expr_tensor.shape[1])
            expr_tensor = torch.cat([expr_tensor, padding], dim=1)
        
        # Forward pass
        outputs = model(dna_tokens, expr_tensor)
        site_probs = outputs['site_probs'][0, :50, 1].cpu().numpy()  # First 50 positions
        
    print("   Position:    ", end="")
    for i in range(50):
        print(f"{i:5d}", end="")
    print()
    
    print("   Probability: ", end="")
    for prob in site_probs:
        print(f"{prob:5.2f}", end="")
    print()
    
    print("   Sequence:    ", end="")
    for i, base in enumerate(sequence[:50]):
        print(f"    {base}", end="")
    print()
    
    # Show thresholding
    binary_sites = (site_probs > 0.5).astype(int)
    print("   Binary(>0.5):", end="")
    for binary in binary_sites:
        print(f"    {binary}", end="")
    print()
    
    print(f"\n💡 KEY INSIGHTS:")
    print("1. Transformer gives PROBABILITIES, not discrete states")
    print("2. You can adjust threshold (0.5) to get more/fewer sites")
    print("3. Higher probabilities = more confident predictions")
    print("4. Can also predict TF type and binding energy simultaneously")
    print("5. Uses full sequence context, not just local windows")


def show_data_input_format():
    """Show how transformer uses the same input files as HMM"""
    
    print(f"\n📁 INPUT DATA FORMAT (Same as your HMM!)")
    print("="*50)
    
    print("The transformer uses EXACTLY the same input files:")
    print("✅ efastanotw12.txt  - DNA sequences (FASTA format)")
    print("✅ expre12.tab       - Expression data (tab-delimited)")
    print("✅ factordts.wtmx    - PWM motifs (your HMM format)")
    
    print("\nNo config.ini needed! Uses command-line arguments:")
    print("  --data_dir R0315/iData")
    print("  --threshold 0.5")
    print("  --model_path checkpoints/best_model.pt")
    
    print(f"\n🔧 PROCESSING PIPELINE:")
    print("1. Load same data files as seq2exp")
    print("2. Convert DNA to tokens: A=0, C=1, G=2, T=3")
    print("3. Embed expression data to same dimension as DNA")
    print("4. Apply attention across full sequence length")
    print("5. Output probabilities for each position")


if __name__ == "__main__":
    show_transformer_output()
    show_data_input_format()