# CSF-Net: Collaborative Spatial-channel attention for Feature fusion Network

[![License: MIT]( https://img.shields.io/badge/License-MIT-yellow.svg) ](LICENSE)
[![Python 3.9+]( https://img.shields.io/badge/Python-3.9+-blue.svg) ]( https://www.python.org/)
[![PyTorch 1.12+]( https://img.shields.io/badge/PyTorch-1.12+-red.svg) ]( https://pytorch.org/)

This is the official implementation of the paper:  
**"Multimodal Brain Tumor MRI Segmentation based on Attention Mechanism and Contrastive Learning"**.

CSF-Net addresses three key challenges in brain tumor segmentation: ambiguous boundaries, suboptimal multimodal fusion, and class imbalance. The core innovations are the **Collaborative Spatial-Channel Attention Module (CSAM)** and the **Feature Fusion Contrastive Loss (FFCL)**.

## 📋 1. Requirements

- Python 3.9+
- PyTorch 1.12.1+
- CUDA 11.7+ (for GPU acceleration)

Install dependencies:
```bash
pip install -r requirements.txt
