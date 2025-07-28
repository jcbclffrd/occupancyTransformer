"""
Main Script for TF Binding Transformer
Demonstrates the complete pipeline from HMM data to transformer predictions

Usage:
    python main.py --mode train     # Train the transformer model
    python main.py --mode predict   # Run inference on sequences
    python main.py --mode compare   # Compare with HMM results
"""

import argparse
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json

from transformer_model import ModelConfig, create_model
from data_preprocessing import create_data_loaders, HMMDataLoader
from training_pipeline import ModelTrainer, BindingSitePredictor, load_trained_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(args):
    """Train the transformer model"""
    logger.info("Starting model training...")
    
    # Configuration
    config = ModelConfig(
        max_seq_length=args.max_seq_length,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        dropout=args.dropout
    )
    
    # Save config
    config_path = Path(args.save_dir) / "config.json"
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Create model
    model = create_model(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    try:
        train_loader, val_loader = create_data_loaders(
            args.data_dir, config, batch_size=config.batch_size
        )
        logger.info(f"Loaded {len(train_loader.dataset)} training samples, "
                   f"{len(val_loader.dataset)} validation samples")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # Create trainer
    trainer = ModelTrainer(model, config, use_wandb=args.use_wandb)
    
    # Train model
    history = trainer.train(train_loader, val_loader, num_epochs=args.epochs, save_dir=args.save_dir)
    
    # Save training history
    history_path = Path(args.save_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {k: [float(x) for x in v] for k, v in history.items()}
        json.dump(history_json, f, indent=2)
    
    logger.info(f"Training completed! Best model saved to {args.save_dir}/best_model.pt")


def predict_sequences(args):
    """Run inference on sequences"""
    logger.info("Running sequence prediction...")
    
    # Load trained model
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    model = load_trained_model(str(model_path))
    predictor = BindingSitePredictor(model)
    
    # Load data
    data_loader = HMMDataLoader(args.data_dir)
    sequences = data_loader.load_sequences("efastanotw12.txt")
    expression_data = data_loader.load_expression_data("expre12.tab")
    
    if not sequences or expression_data.empty:
        logger.error("Failed to load sequence or expression data")
        return
    
    logger.info(f"Loaded {len(sequences)} sequences for prediction")
    
    # Run predictions
    predictions = predictor.predict_batch(
        sequences, expression_data, threshold=args.threshold
    )
    
    # Format results
    results = []
    for seq_id, sites in predictions.items():
        for site in sites:
            results.append({
                'sequence_id': seq_id,
                'position': site['position'],
                'tf_type': site['tf_type'],
                'probability': site['probability'],
                'binding_energy': site['binding_energy'],
                'sequence_window': site['sequence_window']
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = Path(args.output_dir) / "transformer_predictions.csv"
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Found {len(results)} binding sites across {len(predictions)} sequences")
    
    # Print summary statistics
    if len(results) > 0:
        logger.info(f"Average probability: {results_df['probability'].mean():.3f}")
        logger.info(f"Average binding energy: {results_df['binding_energy'].mean():.3f}")
        logger.info(f"Sites per sequence: {len(results) / len(sequences):.1f}")


def compare_methods(args):
    """Compare transformer predictions with HMM results"""
    logger.info("Comparing transformer with HMM predictions...")
    
    # Load transformer model and run predictions
    model = load_trained_model(args.model_path)
    predictor = BindingSitePredictor(model)
    
    # Load data
    data_loader = HMMDataLoader(args.data_dir)
    sequences = data_loader.load_sequences("efastanotw12.txt")
    expression_data = data_loader.load_expression_data("expre12.tab")
    
    # Get transformer predictions
    transformer_results = predictor.compare_with_hmm(sequences, expression_data)
    
    # Save comparison
    output_path = Path(args.output_dir) / "method_comparison.csv"
    output_path.parent.mkdir(exist_ok=True)
    transformer_results.to_csv(output_path, index=False)
    
    logger.info(f"Comparison results saved to {output_path}")


def demonstrate_usage():
    """Demonstrate the complete pipeline with sample data"""
    logger.info("Running demonstration with sample data...")
    
    # Check if data directory exists
    data_dir = Path("R0315/iData")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please ensure your HMM data is in the R0315/iData directory")
        return
    
    # Create small config for demo
    config = ModelConfig(
        max_seq_length=512,
        num_layers=2,
        hidden_dim=128,
        num_heads=4,
        learning_rate=1e-3,
        batch_size=8
    )
    
    try:
        # Load and display sample data
        data_loader = HMMDataLoader(str(data_dir))
        sequences = data_loader.load_sequences("efastanotw12.txt")
        expression_data = data_loader.load_expression_data("expre12.tab")
        motifs = data_loader.load_motif_data("factordts.wtmx")
        
        logger.info(f"Sample data loaded:")
        logger.info(f"  Sequences: {len(sequences)}")
        logger.info(f"  Expression conditions: {expression_data.shape[1] if not expression_data.empty else 0}")
        logger.info(f"  Motifs: {len(motifs)}")
        
        # Show sample sequence
        if sequences:
            sample_id = list(sequences.keys())[0]
            sample_seq = sequences[sample_id]
            logger.info(f"  Sample sequence '{sample_id}': {len(sample_seq)} bp")
            logger.info(f"    First 100 bp: {sample_seq[:100]}")
        
        # Create small model for demo
        model = create_model(config)
        logger.info(f"Demo model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Try to create data loaders
        train_loader, val_loader = create_data_loaders(str(data_dir), config, batch_size=4)
        logger.info(f"Data loaders created: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
        
        # Show a sample batch
        for batch in train_loader:
            logger.info(f"Sample batch shapes:")
            logger.info(f"  DNA tokens: {batch['dna_tokens'].shape}")
            logger.info(f"  Expression: {batch['expression'].shape}")
            logger.info(f"  Site labels: {batch['site_labels'].shape}")
            break
        
        logger.info("Demo completed successfully! Your data is compatible with the transformer pipeline.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        logger.info("Please check your data files and format.")


def main():
    parser = argparse.ArgumentParser(description="TF Binding Transformer Pipeline")
    
    # Common arguments
    parser.add_argument('--mode', choices=['train', 'predict', 'compare', 'demo'], 
                       default='demo', help='Mode to run')
    parser.add_argument('--data_dir', default='R0315/iData', 
                       help='Directory containing HMM data files')
    parser.add_argument('--save_dir', default='checkpoints', 
                       help='Directory to save model checkpoints')
    parser.add_argument('--output_dir', default='results', 
                       help='Directory to save results')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--use_wandb', action='store_true', 
                       help='Use Weights & Biases for logging')
    
    # Model arguments
    parser.add_argument('--max_seq_length', type=int, default=1024, 
                       help='Maximum sequence length')
    parser.add_argument('--num_layers', type=int, default=4, 
                       help='Number of transformer layers')
    parser.add_argument('--hidden_dim', type=int, default=256, 
                       help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=8, 
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='Dropout rate')
    
    # Prediction arguments
    parser.add_argument('--model_path', default='checkpoints/best_model.pt', 
                       help='Path to trained model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Probability threshold for site detection')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'predict':
        predict_sequences(args)
    elif args.mode == 'compare':
        compare_methods(args)
    elif args.mode == 'demo':
        demonstrate_usage()


if __name__ == "__main__":
    main()