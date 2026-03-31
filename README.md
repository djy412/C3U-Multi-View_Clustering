# Contrastive Common and Unique Deep Embedded Clustering
PyTorch implementation of a **multi-view deep clustering framework** that separates each input view into:

- a **common latent representation** for shared semantic structure, and
- a **unique latent representation** for view-specific or complementary information.

The model combines **reconstruction**, **cross-view contrastive learning**, and **prototype-based clustering** to learn cluster-friendly shared embeddings while preserving view-specific detail.

## Overview

Many multi-view learning problems contain two kinds of information:

1. **Shared information** that is consistent across views
2. **Unique information** that is specific to a particular view

This repository implements a model that explicitly decomposes each view into these two parts:

- **Common latent (`c`)**: intended to capture cluster-relevant, view-invariant semantics
- **Unique latent (`u`)**: intended to capture view-specific variation

The framework supports multiple views, configurable latent dimensions, and GPU-based training. It includes utilities for visualization, clustering evaluation, and running experiments on several image-based and multi-view datasets.

## Main Features

- **Multi-view clustering**
- **Separate common and unique latent spaces**
- **Contrastive alignment on the common latent space**
- **Reconstruction from concatenated common + unique latents**
- **Prototype / cluster-center learning in common space**
- **Support for variable number of views**
- **GPU-friendly streaming evaluation utilities**
- **Configurable backbone selection**
- **Visualization tools for embeddings, reconstructions, and training curves**
- **Optional hyperparameter optimization tools via Optuna**

## Method Summary

The model is trained in two main stages:

### 1. Contrastive pretraining
Each view is passed through an encoder-decoder pipeline:

- encoder for **common latent**
- encoder for **unique latent**
- decoder reconstructing the original input from `[c ; u]`

During pretraining, the model minimizes:

- **Reconstruction loss** across all views
- **NT-Xent contrastive loss** between the common latents of different views

This encourages:
- common latents to align across views
- unique latents to preserve the remaining view-specific information needed for reconstruction

### 2. Prototype-based clustering
Cluster centers are learned in the **common latent space**.  
At inference, the model evaluates cluster assignments by combining:

- learned common-space prototypes
- unique latent for each view
- reconstruction error across candidate clusters

The multi-view prediction rule selects the cluster whose prototype + unique latent yields the best normalized reconstruction fit across views.

Example Workflow
Set dataset_name in scripts/config.py
Set the number of views with views
Choose latent dimensions:
LATENT_DIM_C
LATENT_DIM_U
Set training hyperparameters:
LR
PRE_TRAIN_EPOCHS
TEMPERATURE_CON
LAMBDA
Run the script
Inspect saved weights and generated visualizations

Core dependencies
Python 3.9+
PyTorch
torchvision
numpy
pandas
matplotlib
scikit-learn
tqdm
pillow
scikit-image
seaborn
optuna

## Repository Structure
```text
.
├── classes/
│   ├── ResNet18_encoder.py
│   ├── ResNet18_decoder.py
│   └── ResNet18_autoencoder.py
├── scripts/
│   ├── config.py
│   ├── data_loading.py
│   └── utils.py
├── Visualizations/
│   └── Visualization.py
├── weights/
├── main.py
└── README.md

