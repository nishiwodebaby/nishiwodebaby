#!/usr/bin/env python3
"""
Main testing/inference script for CSF-Net.
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CSFNet
from src.data_loader import BraTSDataset
from src.evaluator import Evaluator
from src.utils import load_config, set_seed, get_device, visualize_sample

def main():
    parser = argparse.ArgumentParser(description="Test CSF-Net")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save predictions')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualizations of predictions')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_seed(config['training']['seed'])

    # Get device
    device = get_device(config)

    # Create model
    model = CSFNet(config).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Create test dataset (using validation split for testing)
    from src.data_loader import get_data_loaders
    _, test_loader, _, test_size = get_data_loaders(config)
    
    # For true testing, you would load a separate test set
    # Here we use validation set as an example

    # Create evaluator
    evaluator = Evaluator(config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

    # Inference
    model.eval()
    all_preds = []
    all_targets = []
    patient_slice_info = []

    print(f"Running inference on {test_size} samples...")
    with torch.no_grad():
        for i, (images, targets, patient_ids, slice_ids) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            pred_logits = model(images)
            pred_probs = torch.softmax(pred_logits, dim=1)
            preds = torch.argmax(pred_probs, dim=1)

            # Save predictions and targets
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Save patient and slice info
            for j in range(len(patient_ids)):
                patient_slice_info.append({
                    'patient': patient_ids[j],
                    'slice': slice_ids[j].item()
                })

            # Visualize first few samples
            if args.visualize and i < 10:
                for j in range(len(images)):
                    vis_path = vis_dir / f'{patient_ids[j]}_slice{slice_ids[j].item()}.png'
                    visualize_sample(
                        images[j], targets[j], preds[j],
                        save_path=str(vis_path)
                    )

    # Convert to numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluator.evaluate(all_preds, all_targets)
    evaluator.print_metrics(metrics, title="Test Results")

    # Save predictions and metrics
    np.save(output_dir / 'predictions.npy', all_preds)
    np.save(output_dir / 'targets.npy', all_targets)
    
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main()
