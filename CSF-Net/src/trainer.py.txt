import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        self.criterion = CombinedLoss(config)

        # Optimizer
        train_cfg = config['training']
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg['initial_lr'],
            weight_decay=train_cfg['weight_decay']
        )

        # Learning rate scheduler
        if train_cfg['lr_scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg['num_epochs']
            )
        else:
            self.scheduler = None

        self.num_epochs = train_cfg['num_epochs']
        self.eval_freq = config['evaluation']['eval_freq']
        self.save_freq = config['evaluation']['save_freq']
        self.accumulation_steps = train_cfg.get('accumulation_steps', 1)

        # Metrics tracking
        self.best_dice = 0.0
        self.train_losses = []
        self.val_metrics = []

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_ce = 0.0
        total_ffcl = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        self.optimizer.zero_grad()

        for batch_idx, (images, targets, _, _) in enumerate(pbar):
            images = images.to(self.device)  # [B, 4, H, W]
            targets = targets.to(self.device)  # [B, H, W]

            # Forward pass with feature return for FFCL
            pred_logits, features = self.model(images, return_features=True)

            # Flatten labels for FFCL
            B, C, H, W = pred_logits.shape
            labels_flat = targets.view(B, -1)  # [B, H*W]

            # Compute loss
            loss, dice_loss, ce_loss, ffcl_loss = self.criterion(
                pred_logits, targets, features, labels_flat
            )

            # Gradient accumulation
            loss = loss / self.accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item() * self.accumulation_steps
            total_dice += dice_loss
            total_ce += ce_loss
            total_ffcl += ffcl_loss

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item() * self.accumulation_steps:.4f}',
                'Dice': f'{dice_loss:.4f}',
                'CE': f'{ce_loss:.4f}',
                'FFCL': f'{ffcl_loss:.4f}'
            })

        # Last step if accumulation steps don't align
        if len(self.train_loader) % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        avg_dice = total_dice / len(self.train_loader)
        avg_ce = total_ce / len(self.train_loader)
        avg_ffcl = total_ffcl / len(self.train_loader)

        self.train_losses.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'dice': avg_dice,
            'ce': avg_ce,
            'ffcl': avg_ffcl,
            'lr': self.optimizer.param_groups['lr']
        })

        return avg_loss, avg_dice, avg_ce, avg_ffcl

    @torch.no_grad()
    def validate(self, epoch, evaluator):
        self.model.eval()
        all_preds = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=f'Validation Epoch {epoch+1}')
        for images, targets, patient_ids, slice_ids in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass (no features needed for validation)
            pred_logits = self.model(images)
            pred_probs = torch.softmax(pred_logits, dim=1)
            preds = torch.argmax(pred_probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        metrics = evaluator.evaluate(all_preds, all_targets)
        metrics['epoch'] = epoch + 1

        # Save best model
        if metrics['avg_dice'] > self.best_dice:
            self.best_dice = metrics['avg_dice']
            self.save_checkpoint(epoch, is_best=True)

        self.val_metrics.append(metrics)
        return metrics

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_dice': self.best_dice,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model separately
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"\nSaved best model with Dice: {self.best_dice:.4f}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_dice = checkpoint['best_dice']
        return checkpoint['epoch']

    def train(self, evaluator):
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.num_epochs):
            # Train one epoch
            train_loss, train_dice, train_ce, train_ffcl = self.train_epoch(epoch)

            # Validate if needed
            if (epoch + 1) % self.eval_freq == 0 or epoch == self.num_epochs - 1:
                val_metrics = self.validate(epoch, evaluator)

                print(f"\nEpoch {epoch+1}/{self.num_epochs}")
                print(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, "
                      f"CE: {train_ce:.4f}, FFCL: {train_ffcl:.4f}")
                print(f"Val Metrics - Dice: {val_metrics['avg_dice']:.4f}, "
                      f"Sens: {val_metrics['avg_sensitivity']:.4f}, "
                      f"HD95: {val_metrics['avg_hd95']:.4f}")

            # Save checkpoint periodically
            if (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(epoch, is_best=False)

        print(f"\nTraining completed. Best validation Dice: {self.best_dice:.4f}")
        return self.train_losses, self.val_metrics
