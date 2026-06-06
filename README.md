# C3U-MVC: Contrastive-Complementary Common and Unique Multi-View Clustering

A PyTorch implementation of a **multi-view deep clustering framework** that learns disentangled **shared** and **view-specific** latent representations for unsupervised clustering.

This repository implements a model that decomposes each input view into:

- a **common latent** capturing view-invariant semantic structure, and
- a **unique latent** capturing complementary, view-specific variation.

The framework combines **reconstruction**, **cross-view contrastive alignment**, and **prototype-based clustering** to learn cluster-friendly shared embeddings while preserving the information needed to reconstruct heterogeneous inputs.

---

## Highlights

- **Multi-view deep clustering** with configurable number of views
- **Common / unique latent decomposition**
- **Contrastive learning** applied to the shared latent space
- **Reconstruction from concatenated common + unique embeddings**
- **Prototype-conditioned clustering in common latent space**
- **GPU-friendly streaming utilities** for encoding and evaluation
- **Flexible backbone selection**
- **Visualization tools** for reconstructions, embeddings, losses, and clustering behavior
- **Support for multiple benchmark datasets**

---

## Method Overview

Many multi-view problems contain two fundamentally different types of information:

1. **Shared information** that is consistent across views and useful for clustering
2. **View-specific information** that reflects nuisance factors, complementary detail, or modality-dependent variation

This repository models that structure explicitly.

For each view \(x^{(v)}\), the encoder produces:

- **common latent** \(c^{(v)}\)
- **unique latent** \(u^{(v)}\)

The decoder reconstructs the input from the concatenated latent:

\[
\hat{x}^{(v)} = g([c^{(v)} \oplus u^{(v)}])
\]

### Training proceeds in two stages

#### Stage I: Contrastive pretraining
The model is trained with:

- **reconstruction loss** over all views
- **NT-Xent contrastive loss** between shared latents across views

This encourages the shared representation to align across views while still allowing the unique branch to preserve complementary information.

#### Stage II: Prototype-guided clustering
Cluster centers are learned in the **shared latent space**.  
Cluster assignment is determined by evaluating how well a shared prototype, combined with the view-specific latent, reconstructs the observed inputs across views.

This design encourages prototypes to capture semantic structure in the common latent space while leaving view-dependent detail to the unique branch.

---

## Architecture

The code currently supports a decomposition of the form:

- **Encoder for common latent** \(c\)
- **Encoder for unique latent** \(u\)
- **Decoder** operating on \([c;u]\)
- **Cluster layer** defined in the common latent space

Implemented model components include:

- `CAE` — convolutional autoencoder backbone
- `FCAE` — fully connected autoencoder variant
- `C_U_Model` — top-level clustering model
- `NTXentOnC` — contrastive loss on shared latents
- `Cluster_loss_SoftPullAssigned` — prototype pull + reconstruction refinement loss

Citation:
@article{yates2026c3umvc,
  title={C3U-MVC: ...},
  author={Yates, Don Jared et al.},
  journal={...},
  year={2026}
}

License:
MIT License (see LICENSE file)

## Installation
```bash
git clone https://github.com/djy412/C3U-Multi-View_Clustering.git
cd C3U-Multi-View_Clustering
pip install -r requirements.txt

Reproducing Paper Results
All results in the paper were produced with:

Python 3.10+, PyTorch 2.0+
Adam optimizer (lr=0.001), batch size 64
Fixed random seeds (see config files)

Exact commands and seeds are provided in the scripts/ folder and paper supplementary.

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
