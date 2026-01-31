#!/usr/bin/env python3
"""
Main training script for CSF-Net.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CSFNet
from src.data_loader import get_data_loaders
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.utils import load_config, create_experiment_dir, set_seed, get_device

def main():
    parser = argparse.ArgumentParser(description="Train CSF-Net")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (smaller dataset)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed for reproducibility
    set_seed(config['training']['seed'])

    # Create experiment directory
    exp_dir, config = create_experiment_dir(config)
    print(f"Experiment directory: {exp_dir}")

    # Get device
    device = get_device(config)

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, train_size, val_size = get_data_loaders(config)
    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Create model
    print("Creating model...")
    model = CSFNet(config).to(device)

    # Create evaluator
    evaluator = Evaluator(config)

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)

    # Resume training if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)

    # Train
    train_losses, val_metrics = trainer.train(evaluator)

    # Save final metrics
    import pandas as pd
    train_df = pd.DataFrame(train_losses)
    val_df = pd.DataFrame(val_metrics)
    train_df.to_csv(os.path.join(exp_dir, 'train_metrics.csv'), index=False)
    val_df.to_csv(os.path.join(exp_dir, 'val_metrics.csv'), index=False)

    print(f"\nTraining completed. Results saved to: {exp_dir}")

if __name__ == '__main__':
    main()
