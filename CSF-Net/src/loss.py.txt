import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation."""
    def __init__(self, epsilon=1e-7):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: [B, C, H, W] (logits before softmax)
            target: [B, H, W] (long tensor with class indices)
        """
        pred_probs = F.softmax(pred_logits, dim=1)
        num_classes = pred_probs.shape

        dice_loss = 0.0
        for cls in range(1, num_classes):  # Skip background (class 0)
            pred_cls = pred_probs[:, cls, :, :]
            target_cls = (target == cls).float()

            intersection = torch.sum(pred_cls * target_cls)
            union = torch.sum(pred_cls) + torch.sum(target_cls)

            dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss += 1 - dice

        return dice_loss / (num_classes - 1)

class FFCLoss(nn.Module):
    """
    Feature Fusion Contrastive Loss.
    Pixel-level contrastive loss applied to bottleneck features.
    """
    def __init__(self, temperature=0.1):
        super(FFCLoss, self).__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, features, labels):
        """
        Args:
            features: [B, N, D] where N=H*W, D=embed_dim (flattened bottleneck features)
            labels: [B, H, W] flattened to [B, N] (same spatial positions as features)
        Returns:
            contrastive_loss: Scalar loss value
        """
        B, N, D = features.shape

        # Reshape labels to match features
        labels_flat = labels.view(B, N)  # [B, N]

        # Normalize features
        features_norm = F.normalize(features, dim=-1)  # [B, N, D]

        # Calculate similarity matrix
        sim_matrix = torch.matmul(
            features_norm.view(B * N, D),
            features_norm.view(B * N, D).T
        )  # [B*N, B*N]
        sim_matrix = sim_matrix.view(B, N, B, N)  # [B, N, B, N]
        sim_matrix = sim_matrix / self.temperature

        # Create mask for positive pairs (same class)
        labels_expanded_i = labels_flat.unsqueeze(1).unsqueeze(3)  # [B, 1, N, 1]
        labels_expanded_j = labels_flat.unsqueeze(0).unsqueeze(2)  # [1, B, 1, N]
        positive_mask = (labels_expanded_i == labels_expanded_j) & \
                       ~torch.eye(B, device=features.device).bool().unsqueeze(1).unsqueeze(3)
        positive_mask = positive_mask.float()

        # Mask for negative pairs (different classes)
        negative_mask = (labels_expanded_i != labels_expanded_j).float()

        # For each anchor, denominator is sum over all negatives (and itself)
        # We'll use log-sum-exp trick for numerical stability
        max_sim, _ = torch.max(sim_matrix, dim=(2, 3), keepdim=True)
        sim_matrix_exp = torch.exp(sim_matrix - max_sim)

        # Sum over negatives
        neg_sum = torch.sum(sim_matrix_exp * negative_mask, dim=(2, 3))

        # Sum over positives (excluding self)
        pos_sum = torch.sum(sim_matrix_exp * positive_mask, dim=(2, 3))

        # Compute loss
        losses = -torch.log(pos_sum / (neg_sum + 1e-8))  # [B, N]
        valid_positives = positive_mask.sum(dim=(2, 3)) > 0  # [B, N]

        # Only compute loss where there are positive pairs
        loss = (losses * valid_positives.float()).sum() / (valid_positives.float().sum() + 1e-8)

        return loss

class CombinedLoss(nn.Module):
    """Combination of Dice, CE, and FFCL losses."""
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        loss_cfg = config['loss']
        self.dice_weight = loss_cfg['dice_weight']
        self.ce_weight = loss_cfg['ce_weight']
        self.ffcl_weight = loss_cfg['ffcl_weight']

        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        if self.ffcl_weight > 0:
            self.ffcl_loss = FFCLoss(temperature=loss_cfg['ffcl_temperature'])

    def forward(self, pred_logits, target, features=None, labels_flat=None):
        """
        Args:
            pred_logits: [B, C, H, W]
            target: [B, H, W]
            features: [B, N, D] (optional, for FFCL)
            labels_flat: [B, N] (optional, for FFCL)
        """
        # Base losses
        dice_loss = self.dice_loss(pred_logits, target)
        ce_loss = self.ce_loss(pred_logits, target)

        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        # FFCL loss if applicable
        if self.ffcl_weight > 0 and features is not None and labels_flat is not None:
            ffcl_loss = self.ffcl_loss(features, labels_flat)
            total_loss += self.ffcl_weight * ffcl_loss
            return total_loss, dice_loss.item(), ce_loss.item(), ffcl_loss.item()

        return total_loss, dice_loss.item(), ce_loss.item(), 0.0
