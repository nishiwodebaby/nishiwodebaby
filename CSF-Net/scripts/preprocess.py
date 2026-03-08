#!/usr/bin/env python3
"""
BraTS Dataset Preprocessing Script for CSF-Net
This script preprocesses the raw 3D multimodal BraTS NIFTI files into
2D slices, applies normalization, and saves them as NumPy arrays for efficient training.
参考了医学图像预处理中针对多模态和类别不平衡问题的常见方法[6](@ref)[7](@ref)[8](@ref)。
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Tuple, Dict
import yaml
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import set_seed

def normalize_modality(slice_data: np.ndarray, bottom_percentile: float = 99.0, top_percentile: float = 1.0) -> np.ndarray:
    """
    对单个模态的2D切片进行标准化（Z-score），并裁剪极端值。
    此方法有助于处理不同模态间的对比度差异，是医学图像预处理的常见步骤[6](@ref)。
    
    Args:
        slice_data: 2D numpy array of a single modality.
        bottom_percentile: 上百分位数，用于裁剪高异常值。
        top_percentile: 下百分位数，用于裁剪低异常值。
    
    Returns:
        Normalized 2D slice.
    """
    # 1. 裁剪极端值（类似“去掉最低分和最高分”）
    b = np.percentile(slice_data, bottom_percentile)
    t = np.percentile(slice_data, top_percentile)
    slice_clipped = np.clip(slice_data, t, b)
    
    # 2. 仅对非零区域（即脑组织区域）进行Z-score标准化
    #    避免背景（黑色区域）影响均值和标准差的计算[6](@ref)
    nonzero_mask = slice_clipped > 0
    if np.sum(nonzero_mask) == 0:
        return slice_clipped  # 全背景切片，直接返回
    
    nonzero_region = slice_clipped[nonzero_mask]
    mean_val = np.mean(nonzero_region)
    std_val = np.std(nonzero_region)
    
    if std_val < 1e-7:  # 避免除零
        return slice_clipped
    
    slice_normalized = slice_clipped.copy()
    slice_normalized[nonzero_mask] = (nonzero_region - mean_val) / std_val
    
    # 3. 将背景区域设置为一个固定的低值（如-9），以便在训练中区分
    slice_normalized[~nonzero_mask] = -9.0
    
    return slice_normalized

def crop_to_nonzero_region(volume: np.ndarray, label: np.ndarray = None) -> Tuple[np.ndarray, ...]:
    """
    将3D体积裁剪到包含所有非零体素的最小边界框。
    这可以显著减小数据尺寸，同时保留关键信息，是处理大医学图像时的有效策略[7](@ref)[8](@ref)。
    
    Args:
        volume: 4D numpy array [H, W, D, C] or 3D array [H, W, D].
        label: (Optional) 3D label array [H, W, D] to crop accordingly.
    
    Returns:
        Cropped volume (and label if provided).
    """
    if len(volume.shape) == 4:
        # 多模态情况：合并所有模态的非零掩码
        nonzero_mask = np.any(volume > 0, axis=-1)
    else:
        nonzero_mask = volume > 0
    
    if np.sum(nonzero_mask) == 0:
        return volume, label if label is not None else volume
    
    # 找到非零体素的边界
    nonzero_coords = np.where(nonzero_mask)
    min_coords = np.min(nonzero_coords, axis=1)
    max_coords = np.max(nonzero_coords, axis=1)
    
    # 添加少量边界（如5个体素）以确保上下文信息
    padding = 5
    min_coords = np.maximum(min_coords - padding, 0)
    max_coords = np.minimum(max_coords + padding, np.array(volume.shape[:3]) - 1)
    
    # 执行裁剪
    slices = tuple(slice(min_coords[i], max_coords[i] + 1) for i in range(3))
    
    if len(volume.shape) == 4:
        cropped_volume = volume[slices[0], slices[1], slices[2], :]
    else:
        cropped_volume = volume[slices[0], slices[1], slices[2]]
    
    if label is not None:
        cropped_label = label[slices[0], slices[1], slices[2]]
        return cropped_volume, cropped_label
    
    return cropped_volume

def process_single_patient(patient_dir: Path, output_dir: Path, modalities: List[str], 
                          slice_axis: int = 2, save_2d: bool = True, 
                          skip_empty_slices: bool = True) -> Dict[str, int]:
    """
    处理单个患者的全部数据：加载、裁剪、标准化、切片并保存。
    
    Args:
        patient_dir: Path to patient folder (e.g., BraTS2024_001).
        output_dir: Root directory for preprocessed data.
        modalities: List of modality suffixes.
        slice_axis: Axis along which to extract 2D slices (0=sagittal, 1=coronal, 2=axial).
        save_2d: If True, save individual 2D slices; else save whole 3D volume.
        skip_empty_slices: If True, skip slices with no tumor (for training efficiency).
    
    Returns:
        Dictionary with counts of processed slices.
    """
    patient_id = patient_dir.name
    patient_output_dir = output_dir / patient_id
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {'total_slices': 0, 'tumor_slices': 0, 'empty_slices_skipped': 0}
    
    try:
        # 1. 加载多模态图像
        multimodal_volumes = []
        for mod in modalities:
            mod_path = patient_dir / f'{patient_id}_{mod}.nii.gz'
            if not mod_path.exists():
                # 尝试其他可能的命名变体
                mod_path = patient_dir / f'{patient_id}_{mod.lower()}.nii.gz'
                if not mod_path.exists():
                    raise FileNotFoundError(f"Modality {mod} not found for {patient_id}")
            
            img = nib.load(mod_path)
            data = img.get_fdata().astype(np.float32)
            multimodal_volumes.append(data)
        
        # 堆叠模态 -> [H, W, D, C=4]
        multimodal_data = np.stack(multimodal_volumes, axis=-1)
        
        # 2. 加载标签（分割掩码）
        label_path = patient_dir / f'{patient_id}_seg.nii.gz'
        if not label_path.exists():
            # 对于测试集，可能没有标签
            label_data = None
        else:
            label_img = nib.load(label_path)
            label_data = label_img.get_fdata().astype(np.uint8)
            # 根据BraTS约定转换标签：1,2,4 -> 1,2,3 (NCR/NE, ED, ET)
            label_data 


