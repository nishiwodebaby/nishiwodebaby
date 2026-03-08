import os
import yaml
import torch
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, save_path):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_experiment_dir(config):
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['project_name']}_{timestamp}"
    exp_dir = Path(config['log_dir']) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Update log_dir in config
    config['log_dir'] = str(exp_dir)
    config['checkpoint_dir'] = str(exp_dir / "checkpoints")
    config['result_dir'] = str(exp_dir / "results")
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "visualizations").mkdir(exist_ok=True)
    
    # Save config
    save_config(config, exp_dir / "config.yaml")
    
    return exp_dir, config

def visualize_sample(image, target, prediction, save_path=None):
    """
    Visualize a sample with all modalities, ground truth, and prediction.
    Args:
        image: [4, H, W] tensor (4 modalities)
        target: [H, W] tensor
        prediction: [H, W] tensor
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Modality names
    mod_names = ['FLAIR', 'T1', 'T1CE', 'T2']
    
    # Display each modality
    for i in range(4):
        axes[0, i].imshow(image[i].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'{mod_names[i]}')
        axes[0, i].axis('off')
    
    # Ground truth
    axes[1, 0].imshow(target.cpu().numpy(), cmap='jet', vmin=0, vmax=3)
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    # Prediction
    axes[1, 1].imshow(prediction.cpu().numpy(), cmap='jet', vmin=0, vmax=3)
    axes[1, 1].set_title('Prediction')
    axes[1, 1].axis('off')
    
    # Overlay
    overlay = np.zeros((*target.shape, 3))
    # Red for false positives (pred but not target)
    overlay[prediction > 0] = [1, 0, 0]
    # Green for true positives (both pred and target)
    mask = (prediction > 0) & (target > 0)
    overlay[mask] = [0, 1, 0]
    # Blue for false negatives (target but not pred)
    mask = (target > 0) & (prediction == 0)
    overlay[mask] = [0, 0, 1]
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Overlay (FP=R, TP=G, FN=B)')
    axes[1, 2].axis('off')
    
    # Dice per class
    axes[1, 3].axis('off')
    dice_scores = []
    for cls in [1, 2, 3]:  # WT, TC, ET
        pred_bin = (prediction == cls).float()
        target_bin = (target == cls).float()
        intersection = torch.sum(pred_bin * target_bin)
        union = torch.sum(pred_bin) + torch.sum(target_bin)
        dice = (2. * intersection) / (union + 1e-7)
        dice_scores.append(dice.item())
    
    axes[1, 3].text(0.1, 0.8, f'WT Dice: {dice_scores[0]:.3f}', fontsize=12)
    axes[1, 3].text(0.1, 0.6, f'TC Dice: {dice_scores[1]:.3f}', fontsize=12)
    axes[1, 3].text(0.1, 0.4, f'ET Dice: {dice_scores[2]:.3f}', fontsize=12)
    axes[1, 3].text(0.1, 0.2, f'Avg Dice: {np.mean(dice_scores):.3f}', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_device(config):
    """Get PyTorch device based on configuration."""
    if config['training']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device
