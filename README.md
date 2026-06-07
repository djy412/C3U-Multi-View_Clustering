# C3U-MVC: Contrastive-Complementary Common and Unique Multi-View Clustering

**Official PyTorch implementation** of the model presented in our paper:

> **Disentangling shared and Specific Features in Deep Multimodal Clustering with Contrastive-Complementary Learning**  
> Don Jared Yates et al.  
> [Paper Link](https://arxiv.org/abs/...) | [GitHub](https://github.com/djy412/C3U-Multi-View_Clustering)

## Overview

C3U-MVC is a deep embedded multi-view clustering framework that disentangles **shared (common)** and **view-specific (unique)** representations *without* imposing conditional independence or orthogonality constraints. The model uses:

- Cross-view contrastive alignment **only** in the shared subspace
- Reconstruction from concatenated shared + view-specific latents to preserve complementary information
- A two-stage training procedure (contrastive pretraining → positives-only prototype refinement)
- Prototype-conditioned reconstruction with per-instance best-view selection for robust clustering

## Features

- Dual-encoder (shared + view-specific), single decoder architecture
- Strong performance on both image-pair and true multi-view benchmarks
- Support for image data and heterogeneous feature vectors (e.g., Caltech-101 subsets)
- Reproducible training scripts with fixed seeds

## Installation

```
bash
git clone https://github.com/djy412/C3U-Multi-View_Clustering.git
cd C3U-Multi-View_Clustering

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Quick Start
1. Edit `scripts/config.py` to select the dataset, latent dimensions, number of views, epochs, and random seed.
2. Run:
```
bash
python main.py
```
For full hyperparameter options:
```
bash
python main.py --help
```
Reproducing Paper Results
All results reported in the paper were obtained using:

Python 3.13.7
torch==2.11.0.dev20260121+cu128
torchvision==0.25.0.dev20260121+cu128
Adam optimizer (lr=0.001), batch size 128
Fixed random seeds (configurable)

See scripts/config.py for dataset-specific settings. Pre-trained weights are available in the weights/ directory.
Repository Structure
```
textC3U-Multi-View_Clustering/
├── classes/                # Core model classes (encoders, decoder, C3U model)
├── scripts/
│   ├── config.py           # Hyperparameters and dataset settings
│   ├── data_loading.py     # Data loaders for multi-view datasets
│   └── utils.py            # Training utilities and metrics
├── Visualizations/         # Visualization and plotting scripts
├── weights/                # Pre-trained model checkpoints
├── main.py                 # Main training script
├── requirements.txt
├── LICENSE
└── README.md
```
Citation
If you use this code or build upon our work, please cite our paper:
```
bibtex
  @inproceedings{yates2026c3umvc,
  title={Disentangling shared and Specific Features in Deep Multimodal Clustering with Contrastive-Complementary Learning},
  author={Yates, Don and Sevil, Hakki Erhan and Mahyari, Arash},
  booktitle={...},
  year={2026}
}
```

License
This project is licensed under the MIT License — see the LICENSE file for details.

Contact
Questions or issues? Feel free to open an issue on GitHub or contact the authors.
