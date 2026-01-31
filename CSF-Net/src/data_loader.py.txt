import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
import yaml
from typing import List, Tuple, Dict

class BraTSDataset(Dataset):
    """
    Dataset for loading 2D slices from BraTS multimodal 3D MRI volumes.
    """
    def __init__(self, data_root: str, patient_ids: List[str], modalities: List[str], 
                 transform=None, is_train: bool = True, slice_axis: int = 2):
        """
        Args:
            data_root: Root directory containing patient folders.
            patient_ids: List of patient folder names.
            modalities: List of modality suffixes (e.g., ['flair', 't1', 't1ce', 't2']).
            transform: Optional transform to apply to the image slice.
            is_train: Whether the dataset is for training.
            slice_axis: Axis along which to extract 2D slices (0, 1, 2 for axial).
        """
        self.data_root = Path(data_root)
        self.patient_ids = patient_ids
        self.modalities = modalities
        self.transform = transform
        self.is_train = is_train
        self.slice_axis = slice_axis

        # Preload file paths for efficiency
        self.samples = []  # List of (patient_id, slice_idx) tuples
        for pid in patient_ids:
            seg_path = self.data_root / pid / f'{pid}_seg.nii.gz'
            if not seg_path.exists():
                continue
            seg_img = nib.load(seg_path)
            seg_data = seg_img.get_fdata()
            # Only include slices that contain tumor (for training efficiency)
            if is_train:
                valid_slices = np.unique(np.where(seg_data > 0)[slice_axis])
            else:
                # For validation, include all slices or every N-th slice
                num_slices = seg_data.shape[slice_axis]
                valid_slices = list(range(0, num_slices, 2))  # Subsample
            for sl in valid_slices:
                self.samples.append((pid, sl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, slice_idx = self.samples[idx]
        patient_dir = self.data_root / pid

        # Load multimodal images
        images = []
        for mod in self.modalities:
            img_path = patient_dir / f'{pid}_{mod}.nii.gz'
            img = nib.load(img_path)
            img_data = img.get_fdata()
            # Extract 2D slice
            if self.slice_axis == 0:
                slice_data = img_data[slice_idx, :, :]
            elif self.slice_axis == 1:
                slice_data = img_data[:, slice_idx, :]
            else:  # axial (default)
                slice_data = img_data[:, :, slice_idx]
            images.append(slice_data)

        # Stack modalities to form a 4-channel 2D image [H, W, C=4]
        multimodal_img = np.stack(images, axis=-1)  # Shape: [H, W, 4]

        # Load and process label
        seg_path = patient_dir / f'{pid}_seg.nii.gz'
        seg = nib.load(seg_path)
        seg_data = seg.get_fdata()
        if self.slice_axis == 0:
            label_slice = seg_data[slice_idx, :, :]
        elif self.slice_axis == 1:
            label_slice = seg_data[:, slice_idx, :]
        else:
            label_slice = seg_data[:, :, slice_idx]

        # Convert to multi-class label: 0=BG, 1=NCR/NE, 2=ED, 4=ET -> 1,2,3
        # According to BraTS convention: label 1,2,4 -> class 1,2,3
        label_slice = np.where(label_slice == 4, 3, label_slice)  # ET -> class 3

        # Normalize image slice (per modality)
        for c in range(multimodal_img.shape[-1]):
            modality_slice = multimodal_img[..., c]
            mean = modality_slice.mean()
            std = modality_slice.std()
            if std > 0:
                multimodal_img[..., c] = (modality_slice - mean) / std
            else:
                multimodal_img[..., c] = modality_slice - mean

        # Convert to PyTorch tensors and reshape to [C, H, W]
        image_tensor = torch.from_numpy(multimodal_img).float().permute(2, 0, 1)  # [4, H, W]
        label_tensor = torch.from_numpy(label_slice).long()  # [H, W]

        if self.transform:
            image_tensor, label_tensor = self.transform(image_tensor, label_tensor)

        return image_tensor, label_tensor, str(pid), slice_idx

def get_data_loaders(config: Dict):
    """
    Create train and validation DataLoaders based on config.
    """
    data_cfg = config['data']
    train_cfg = config['training']

    # Get list of patient folders
    data_root = Path(data_cfg['data_root'])
    all_patients = sorted([d.name for d in data_root.iterdir() if d.is_dir()])

    # Simple train/val split (adjust if official split is available)
    split_idx = int(len(all_patients) * data_cfg.get('train_split', 0.8))
    train_patients = all_patients[:split_idx]
    val_patients = all_patients[split_idx:]

    # Create datasets
    train_dataset = BraTSDataset(
        data_root=data_root,
        patient_ids=train_patients,
        modalities=data_cfg['modalities'],
        is_train=True,
        slice_axis=2  # Axial slices
    )
    val_dataset = BraTSDataset(
        data_root=data_root,
        patient_ids=val_patients,
        modalities=data_cfg['modalities'],
        is_train=False,
        slice_axis=2
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Often use batch_size=1 for validation
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, len(train_dataset), len(val_dataset)
