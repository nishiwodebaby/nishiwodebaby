import numpy as np
from scipy.spatial.distance import directed_hausdorff
import torch
import SimpleITK as sitk

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.eval_regions = config['data']['eval_regions']  # [1, 2, 3] for WT, TC, ET
        self.region_names = ['WT', 'TC', 'ET']

    def dice_coefficient(self, pred, target):
        """Calculate Dice coefficient for a binary mask."""
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)
        dice = (2. * intersection) / (union + 1e-7)
        return dice

    def sensitivity(self, pred, target):
        """Calculate sensitivity (recall)."""
        tp = np.sum(pred * target)
        fn = np.sum((1 - pred) * target)
        sens = tp / (tp + fn + 1e-7)
        return sens

    def precision(self, pred, target):
        """Calculate precision."""
        tp = np.sum(pred * target)
        fp = np.sum(pred * (1 - target))
        prec = tp / (tp + fp + 1e-7)
        return prec

    def hausdorff95(self, pred, target):
        """Calculate 95% Hausdorff Distance."""
        if np.sum(pred) == 0 or np.sum(target) == 0:
            return np.nan

        # Get coordinates of non-zero points
        pred_coords = np.argwhere(pred > 0)
        target_coords = np.argwhere(target > 0)

        if len(pred_coords) == 0 or len(target_coords) == 0:
            return np.nan

        # Compute directed Hausdorff distances
        h1 = directed_hausdorff(pred_coords, target_coords)
        h2 = directed_hausdorff(target_coords, pred_coords)
        hd = max(h1, h2)

        # For more robust 95% HD, you would need to compute percentiles
        # This is a simplified version
        return hd

    def evaluate_slice(self, pred_slice, target_slice):
        """Evaluate a single 2D slice."""
        metrics = {}
        for i, region in enumerate(self.eval_regions):
            region_name = self.region_names[i]

            # Create binary masks for this region
            pred_binary = (pred_slice == region).astype(np.float32)
            target_binary = (target_slice == region).astype(np.float32)

            # Compute metrics
            dice = self.dice_coefficient(pred_binary, target_binary)
            sens = self.sensitivity(pred_binary, target_binary)
            prec = self.precision(pred_binary, target_binary)
            hd95 = self.hausdorff95(pred_binary, target_binary)

            metrics[f'{region_name}_dice'] = dice
            metrics[f'{region_name}_sensitivity'] = sens
            metrics[f'{region_name}_precision'] = prec
            metrics[f'{region_name}_hd95'] = hd95

        return metrics

    def evaluate_volume(self, pred_volume, target_volume):
        """Evaluate a 3D volume (aggregate over slices)."""
        all_metrics = {f'{region}_{metric}': [] for region in self.region_names 
                      for metric in ['dice', 'sensitivity', 'precision', 'hd95']}

        for slice_idx in range(pred_volume.shape[0](@ref):
            slice_metrics = self.evaluate_slice(pred_volume[slice_idx], target_volume[slice_idx])
            for key, value in slice_metrics.items():
                if not np.isnan(value):
                    all_metrics[key].append(value)

        # Aggregate over slices
        aggregated = {}
        for key, values in all_metrics.items():
            if len(values) > 0:
                aggregated[key] = float(np.mean(values))
            else:
                aggregated[key] = 0.0

        # Compute average metrics
        aggregated['avg_dice'] = np.mean([aggregated[f'{region}_dice'] for region in self.region_names])
        aggregated['avg_sensitivity'] = np.mean([aggregated[f'{region}_sensitivity'] for region in self.region_names])
        aggregated['avg_precision'] = np.mean([aggregated[f'{region}_precision'] for region in self.region_names])
        aggregated['avg_hd95'] = np.mean([aggregated[f'{region}_hd95'] for region in self.region_names if not np.isnan(aggregated[f'{region}_hd95'])])

        return aggregated

    def evaluate(self, all_preds, all_targets):
        """Evaluate across all samples in validation set."""
        total_metrics = {f'{region}_{metric}': 0.0 for region in self.region_names 
                        for metric in ['dice', 'sensitivity', 'precision', 'hd95']}
        count = 0

        for i in range(len(all_preds)):
            pred = all_preds[i]
            target = all_targets[i]

            # Evaluate this 2D slice
            slice_metrics = self.evaluate_slice(pred, target)

            # Accumulate
            for key, value in slice_metrics.items():
                if not np.isnan(value):
                    total_metrics[key] += value

            count += 1

        # Average across all slices
        for key in total_metrics.keys():
            total_metrics[key] /= count

        # Compute overall averages
        total_metrics['avg_dice'] = np.mean([total_metrics[f'{region}_dice'] for region in self.region_names])
        total_metrics['avg_sensitivity'] = np.mean([total_metrics[f'{region}_sensitivity'] for region in self.region_names])
        total_metrics['avg_precision'] = np.mean([total_metrics[f'{region}_precision'] for region in self.region_names])

        # Handle possible NaN in HD95
        hd95_values = []
        for region in self.region_names:
            val = total_metrics[f'{region}_hd95']
            if not np.isnan(val):
                hd95_values.append(val)
        total_metrics['avg_hd95'] = np.mean(hd95_values) if hd95_values else np.nan

        return total_metrics

    def print_metrics(self, metrics, title="Evaluation Results"):
        """Print formatted metrics."""
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        print(f"{'Region':<10} {'Dice':<10} {'Sens':<10} {'Prec':<10} {'HD95':<10}")
        print(f"{'-'*50}")

        for region in self.region_names:
            dice = metrics.get(f'{region}_dice', 0.0)
            sens = metrics.get(f'{region}_sensitivity', 0.0)
            prec = metrics.get(f'{region}_precision', 0.0)
            hd95 = metrics.get(f'{region}_hd95', 0.0)
            print(f"{region:<10} {dice:.4f}     {sens:.4f}     {prec:.4f}     {hd95:.4f}")

        print(f"{'-'*50}")
        print(f"{'Average':<10} {metrics['avg_dice']:.4f}     "
              f"{metrics['avg_sensitivity']:.4f}     "
              f"{metrics['avg_precision']:.4f}     "
              f"{metrics['avg_hd95']:.4f}")
        print(f"{'='*50}\n")
