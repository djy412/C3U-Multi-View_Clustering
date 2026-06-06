# -*- coding: utf-8 -*-
"""
Contrastive Common and Unique Deep Embedded Clustering
This model varies the Common and Unique latent space
Optimized for GPU
Backbone selection
Hard negative mining
Multi-view support set by views in config file
Created on Wed Jul 10 10:08:42 2024
@author: djy41
"""
from scripts.config import IMG_SIZE, W_PULL, MARGIN, views, NORMALIZED, TEMPERATURE_CON, BATCH_SIZE, PRE_TRAIN_EPOCHS, MODEL_FILENAME, LR, WORKERS, LATENT_DIM_C, LATENT_DIM_U, NUM_CLASSES, LAMBDA, GAMMA, FINE_TUNE_EPOCHS, dataset_name , CHANNELS, TOLERANCE, UPDATE_INTERVAL
from classes.ResNet18_encoder import ResNet18EncoderFC
from classes.ResNet18_decoder import ResNet18DecoderFC
from classes.ResNet18_autoencoder import ResNet18Autoencoder 
from Visualizations.Visualization import Show_settings, Show_dataloader_data, Show_Training_Loss, Show_Component_Embeddings, Show_Componet_Reconstructions, Show_Embedding_Space, Show_Complete_Reconstructions, Show_Partial_Embedding_Space, Show_Results, Show_Representation, Show_NMI_By_Epochs, Show_Variance
from scripts.data_loading import get_cifar10_data_loaders, get_KITTI_Multi_View_dataloaders, load_data, get_MVPN_dataloaders, get_Multi_Market_dataloaders, Multi_Market_Dataset, get_Multi_MNIST_dataloaders, Multi_MNIST_Dataset, get_MVPN_dataloaders, MVPN_Dataset, get_Multi_FASHION_dataloaders, Multi_FASHION_Dataset, get_Multi_STL_10_dataloaders, Multi_STL_10_Dataset
from scripts.utils import cluster_acc2, train_epoch, test_epoch, plot_ae_outputs, cluster_acc, calculate_purity, set_seed 
import torch
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, f1_score, confusion_matrix, precision_score
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm
#--- For hyperparameter optimization
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.trial import TrialState
from optuna_dashboard import run_server

from torch.optim import Adam
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from skimage.metrics import structural_similarity as ssim
import itertools
import seaborn as sns
import pandas as pd

import random
import math
import torch.nn.functional as F
from torch.nn.functional import normalize
import torchvision.models as models

PreTRAIN = True
SEED = 12
ADD_NOISE_EVAL = False  # New flag: Add Gaussian noise to one view during final eval
NOISE_STD = 0.1  # Noise standard deviation for eval test

set_seed(SEED)

BACKBONE = "cae"   # "fcae" | "cae" | "vgg16" | "resnet18"

V = views

#************************************************************************
#--- Define Function collect embeddings for display
#************************************************************************
def collect_plot_embeddings_stream(model, loader, V, device, view_idx=0, max_samples=None):
    """
    Collect c and u for one chosen view across the dataset, batch-by-batch.
    Returns CPU tensors:
        c_all: [N, Dc]
        u_all: [N, Du]
        y_all: [N]
    """
    model.eval()
    c_buf, u_buf, y_buf = [], [], []
    seen = 0

    with torch.no_grad():
        for batch in loader:
            xs, y, _ = unpack_views(batch, V)
            x = xs[view_idx].to(device, non_blocking=True)

            _, c, u = model.ae(x)

            c_buf.append(c.cpu())
            u_buf.append(u.cpu())
            y_buf.append(y.cpu())

            seen += x.size(0)
            if max_samples is not None and seen >= max_samples:
                break

    c_all = torch.cat(c_buf, dim=0)
    u_all = torch.cat(u_buf, dim=0)
    y_all = torch.cat(y_buf, dim=0)

    if max_samples is not None:
        c_all = c_all[:max_samples]
        u_all = u_all[:max_samples]
        y_all = y_all[:max_samples]

    return c_all, u_all, y_all
#----------------------------------------------------------------------------

#************************************************************************
#--- Define Function to not load full dataset
#************************************************************************
def compute_common_features_stream(model, loader, V, device):
    """
    Encodes the dataset batch-by-batch and returns:
      c_all: [N, Dc] on CPU
      y_all: [N] on CPU
    Uses the average common representation across views.
    """
    model.eval()
    c_buf, y_buf = [], []

    with torch.no_grad():
        for batch in loader:
            xs, y, _ = unpack_views(batch, V)
            xs = [x.to(device, non_blocking=True) for x in xs]

            cs = []
            for x in xs:
                _, c, _ = model.ae(x)
                cs.append(c)

            c_mean = torch.stack(cs, dim=0).mean(dim=0)   # [B, Dc]
            c_buf.append(c_mean.cpu())
            y_buf.append(y.cpu())

            # release batch tensors promptly
            del xs, cs, c_mean

    return torch.cat(c_buf, dim=0), torch.cat(y_buf, dim=0)
#----------------------------------------------------------------------------

#************************************************************************
#--- Define Less memory intensitve EVALUATOR
#************************************************************************
def predict_by_center(model, x):
    _, c, _ = model.ae(x)
    centers = model.cluster_layer.data
    d = torch.cdist(c, centers)
    return torch.argmin(d, dim=1)
#----------------------------------------------------------------------------

#************************************************************************
#--- Define Less memory intensitve EVALUATOR
#************************************************************************
def eval_center_stream(model, loader, V, device, view_idx=0):
    yt, yp = [], []
    with torch.no_grad():
        for batch in loader:
            xs, y, _ = unpack_views(batch, V)
            x = xs[view_idx].to(device, non_blocking=True)
            pred = predict_by_center(model, x)
            yt.append(y.cpu().numpy())
            yp.append(pred.cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp)
#----------------------------------------------------------------------------

#************************************************************************
#--- Define EVALUATOR across views
#************************************************************************
def eval_multiview_model(model, loader, V, device):
    yt, yp = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            xs, y, _ = unpack_views(batch, V)
            xs = [x.to(device, non_blocking=True) for x in xs]
            pred = model(*xs)  # <-- your real multi-view decision rule
            yt.append(y.cpu().numpy())
            yp.append(pred.cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp)
#----------------------------------------------------------------------------

#************************************************************************
#--- Define multi-view unpacker
#************************************************************************
def unpack_views(batch, V: int):
    """
    Returns:
        xs: list[Tensor] length V, each [B,C,H,W]
        y:  Tensor [B]
        idx: Tensor [B]
    Supports either:
      (x1, x2, ..., xV, y, idx)  OR  (xs_list, y, idx)
    """
    if isinstance(batch[0], (list, tuple)):
        xs, y, idx = batch
        xs = list(xs)
    else:
        # assume last two are y, idx
        *xs, y, idx = batch
        xs = list(xs)

    if len(xs) != V:
        raise ValueError(f"Expected V={V} views, got {len(xs)}")
    return xs, y, idx
# ----------------------------------------------------------------------------

#************************************************************************
#--- Define Convolutional Auto Encoder with separate C/U sizes
#************************************************************************
class CAE(nn.Module):
    def __init__(self, LATENT_DIM_C: int, LATENT_DIM_U: int):
        super(CAE, self).__init__()
        self.LATENT_DIM_C = LATENT_DIM_C
        self.LATENT_DIM_U = LATENT_DIM_U

        #--- Encoder_Common
        self.encoder_c = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, kernel_size=3, stride=2, padding=1),  # 48x48
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 24x24
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 12x12
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6x6
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512*2*2, LATENT_DIM_C) #--- 32x32 images  
            #nn.Linear(512*4*4, LATENT_DIM_C) #--- 64x64 images  
            #nn.Linear(512*6*6, LATENT_DIM_C) #--- 96x96 images  
            #nn.Linear(512*8*8, LATENT_DIM_C) #--- 128x128 images 
        )
        
        #--- Encoder_Peculiar
        self.encoder_p = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, kernel_size=3, stride=2, padding=1),  # 48x48
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 24x24
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 12x12
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6x6
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512*2*2, LATENT_DIM_U) #--- 32x32 images
            #nn.Linear(512*4*4, LATENT_DIM_U) #--- 64x64 images
            #nn.Linear(512*6*6, LATENT_DIM_U) #--- 96x96 images  
            #nn.Linear(512*8*8, LATENT_DIM_U) #--- 128x128 images
        )

        #--- Decoder (takes [c,u] concat)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM_C+LATENT_DIM_U, 512*2*2), # 32x32
            #nn.Linear(LATENT_DIM_C+LATENT_DIM_U, 512*4*4), # 64x64
            #nn.Linear(LATENT_DIM_C+LATENT_DIM_U, 512*6*6), # 96x96
            #nn.Linear(LATENT_DIM_C+LATENT_DIM_U, 512*8*8), # 128x128
            nn.ReLU(),
            nn.Unflatten(1, (512,2,2)), # 32x32
            #nn.Unflatten(1, (512,4,4)), # 64x64
            #nn.Unflatten(1, (512,6,6)), # 96x96
            #nn.Unflatten(1, (512,8,8)),  # 128x128
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 12x12
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 24x24
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # change padding to 2 for 28x28

            nn.ReLU(),
            nn.ConvTranspose2d(64, CHANNELS, kernel_size=3, stride=2, padding=1, output_padding=1),  # 96x96
            nn.Sigmoid()  # Output in range [0,1]
        )

    def pretrain(self, data_loader):
        pretrain_ae(self, data_loader)

    def forward(self, x):      
        c = self.encoder_c(x)
        u = self.encoder_p(x)
        z = torch.concat([c, u], dim=1)
        x_bar = self.decoder(z)
        return x_bar, c, u
# ----------------------------------------------------------------------------

# ************************************************************************
# --- Define Fully Connected Auto Encoder with separate C/U sizes (modified for arbitrary in_dim, vector inputs)
# ************************************************************************
class FCAE(nn.Module):
    """
    Fully-connected autoencoder with separate common (c) and peculiar (u) latents.
    Modified to handle vector inputs (not images).
    Now operates only on projected (fixed SHARED_DIM) inputs; projectors are handled externally.
    """
    def __init__(
        self,
        LATENT_DIM_C: int,
        LATENT_DIM_U: int,
        shared_encoder_c: nn.Module,
        shared_encoder_u: nn.Module,
        shared_decoder: nn.Module,
        hidden: tuple = (1024, 512, 256)
    ):
        super().__init__()
        self.LATENT_DIM_C = LATENT_DIM_C
        self.LATENT_DIM_U = LATENT_DIM_U
        self.shared_encoder_c = shared_encoder_c
        self.shared_encoder_u = shared_encoder_u
        self.shared_decoder = shared_decoder

    def decoder(self, z):
        x_bar = self.shared_decoder(z)
        return x_bar

    def forward(self, projected):
        c = self.shared_encoder_c(projected)
        u = self.shared_encoder_u(projected)
        z = torch.cat([c, u], dim=1)
        x_bar_shared = self.decoder(z)
        x_bar_shared = torch.sigmoid(x_bar_shared)  # Force [0,1] for BCE stability
        return x_bar_shared, c, u  # Return shared recon; inverse project externally
# ----------------------------------------------------------------------------

#************************************************************************
#--- Define Common and Unique Model (cluster centers in common space)
#************************************************************************
class C_U_Model(nn.Module):
    def __init__(self, LATENT_DIM_C, LATENT_DIM_U, backbone="fcae", img_size=32):
        super().__init__()
        self.alpha = 1.0
        self.LATENT_DIM_C = LATENT_DIM_C
        self.LATENT_DIM_U = LATENT_DIM_U

        if backbone.lower() == "cae":
            self.ae = CAE(LATENT_DIM_C, LATENT_DIM_U)
        if backbone.lower() == "fcae":            
            self.ae = FCAE(LATENT_DIM_C, LATENT_DIM_U)

        self.cluster_layer = Parameter(torch.Tensor(NUM_CLASSES, LATENT_DIM_C))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, data_loader):
        pretrain_ae(self.ae, data_loader)

    def forward(self, *xs):
        """
        xs: variable number of views, each [B,C,H,W]
        returns:
            cluster_est: [B]
            hard_negatives: [B, D_c]
        """
        V_local = len(xs)
        if V_local < 2:
            raise ValueError("Need at least 2 views")

        B = xs[0].size(0)
        K = NUM_CLASSES
        device = xs[0].device

        # --- encode u for each view (peculiar part only)
        us = []
        for x in xs:
            _, _, u = self.ae(x)
            us.append(u)

        centers = self.cluster_layer.unsqueeze(0).expand(B, K, -1)  # [B,K,Dc]

        mse_views = []
        for v in range(V_local):
            u = us[v]
            u_exp = u.unsqueeze(1).expand(B, K, -1)                  # [B,K,Du]
            z = torch.cat([centers, u_exp], dim=2).reshape(B * K, -1)

            rec = self.ae.decoder(z).view(B, K, *xs[v].shape[1:])    # [B,K,C,H,W]
            x_exp = xs[v].unsqueeze(1).expand(-1, K, -1, -1, -1)
            mse = ((rec - x_exp) ** 2).mean(dim=[2, 3, 4])           # [B,K]
            mse_views.append(mse)

        # --- global normalization across all views/classes (like your original)
        all_mse = torch.cat([m.flatten() for m in mse_views], dim=0)
        gmin = all_mse.min()
        gmax = all_mse.max()
        denom = (gmax - gmin).clamp_min(1e-8)

        mse_views_n = [(m - gmin) / denom for m in mse_views]        # list of [B,K]

        # --- min over views per class
        mse_stack = torch.stack(mse_views_n, dim=0)                  # [V,B,K]
        mse_errs = mse_stack.min(dim=0).values                       # [B,K]

        _, k_pos = torch.min(mse_errs, dim=1)                  # [B]

        return k_pos
#----------------------------------------------------------------------------

#************************************************************************
#--- Define NTX Loss 
#************************************************************************
class NTXentOnC(torch.nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.tau = temperature

    def forward(self, c_view1, c_view2):
        B = c_view1.size(0)
        z = torch.cat([c_view1, c_view2], dim=0)             # [2B, D]
        sim = torch.mm(z, z.t()) / self.tau                  # [2B, 2B]
        mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float('-inf'))           # remove self-sim
        # positives: i ↔ i+B and i+B ↔ i
        pos_index = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)  # [2B]
        loss = F.cross_entropy(sim, pos_index)              # row-wise softmax CE
        
        return loss
#----------------------------------------------------------------------------

#************************************************************************
#--- Define Cluster Loss Pull toward prototypes Softly BEST
#************************************************************************
class Cluster_loss_SoftPullAssigned(nn.Module):
    """
    Soft pull with externally assigned per-sample centers.

    mu_batch:   [B, Dc]   assigned center for each sample (e.g. mu_est[0][idx])
    c_list:     list of V tensors, each [B, Dc]
    xhat_list:  list of V tensors, each same shape as x_list
    x_list:     list of V tensors
    """
    def __init__(self, gamma=GAMMA, w_pull=1.0):
        super().__init__()
        self.gamma = float(gamma)
        self.w_pull = float(w_pull)

    def forward(self, mu_batch, c_list, xhat_list, x_list):
        assert mu_batch.dim() == 2, "mu_batch must be [B, Dc]"
        assert len(c_list) == len(xhat_list) == len(x_list), "Inputs must match in length"

        pull_terms = []
        for c in c_list:
            # squared L2 distance to assigned center, per sample
            d2 = torch.sum((c - mu_batch) ** 2, dim=1)   # [B]
            pull_terms.append(d2.mean())

        # average across views
        loss_pull = torch.stack(pull_terms).mean()

        # reconstruction term
        #loss_rec = sum(F.binary_cross_entropy(xh, x) for xh, x in zip(xhat_list, x_list))
        loss_rec = sum(F.mse_loss(xh, x) for xh, x in zip(xhat_list, x_list))
        
        return self.w_pull * loss_pull + self.gamma * loss_rec
#----------------------------------------------------------------------------

#************************************************************************
#--- Define Pretraining Training on Common loss
#************************************************************************
def pretrain_ae(model, data_loader):

    if PreTRAIN:
        optimizer = Adam(model.parameters(), LR)
        con_loss = NTXentOnC(temperature=TEMPERATURE_CON)
        recon = nn.MSELoss()
        #recon = nn.BCELoss()

        Loss_history = []
        for epoch in range(PRE_TRAIN_EPOCHS):
            total = total_rec = total_ntx = 0.0

            for batch in data_loader:
                xs, _, _ = unpack_views(batch, V)        # ignore y, idx in pretrain
                xs = [x.to(device) for x in xs]

                optimizer.zero_grad()

                # forward each view
                xhats, cs, us = [], [], []
                for x in xs:
                    x_hat, c, u = model(x)
                    xhats.append(x_hat)
                    cs.append(F.normalize(c, dim=1))
                    us.append(u)

                # recon: sum over views
                loss_rec = sum(recon(xh, x) for xh, x in zip(xhats, xs))

                # ntx: sum over all unordered pairs (v<w)
                loss_ntx = 0.0
                for i in range(V):
                    for j in range(i + 1, V):
                        loss_ntx = loss_ntx + con_loss(cs[i], cs[j])

                loss = loss_rec + LAMBDA * loss_ntx
                loss.backward()
                optimizer.step()

                total += loss.item()
                total_rec += loss_rec.item()
                total_ntx += loss_ntx.item()

            avg = total / len(data_loader)
            Loss_history.append(avg)
            print(f"epoch {epoch} total_loss={avg:.4f} | rec={total_rec/len(data_loader):.4f} | ntx={total_ntx/len(data_loader):.4f}")

        #--- Plot the training loss
        Show_Training_Loss(Loss_history)
        
        # Compute reconstruction gain Δ^v for all views
        delta_by_view = [[] for _ in range(V)]
        
        with torch.no_grad():
            for batch in data_loader:
                xs, _, _ = unpack_views(batch, V)
                xs = [x.to(device) for x in xs]
        
                for v in range(V):
                    x = xs[v]
                    x_hat, c, u = model(x)
        
                    z_zero = torch.cat([c, torch.zeros_like(u)], dim=1)
                    x_hat_zero = model.decoder(z_zero)
        
                    delta = recon(x_hat_zero, x) - recon(x_hat, x)
                    delta_by_view[v].append(delta.item())
        
        # Print stats
        for v in range(V):
            vals = np.array(delta_by_view[v], dtype=np.float32)
            mean = float(vals.mean()) if vals.size else 0.0
            std  = float(vals.std())  if vals.size else 0.0
            print(f"Average reconstruction gain Δ^{v+1}: {mean:.4f} ± {std:.4f}")
        
        #--- Save the model weights 
        if dataset_name == 'Dataset: Multi-MNIST':
            torch.save(model.state_dict(), 'weights/MNIST_ae.pkl')
        if dataset_name == 'Dataset: Multi-FASHION':
            torch.save(model.state_dict(), 'weights/FASHION_ae.pkl')
        if dataset_name == 'Fashion':
            torch.save(model.state_dict(), 'weights/Fashion_ae.pkl')
        if dataset_name == 'Dataset: Multi-Market':
            torch.save(model.state_dict(), 'weights/Market_ae.pkl')
        if dataset_name == 'Dataset: Multi-MVP-N':
            torch.save(model.state_dict(), 'weights/MVP-N_ae.pkl')
        if dataset_name == 'Dataset: Multi-STL-10':
            torch.save(model.state_dict(), 'weights/STL-10_ae.pkl')
        if dataset_name == 'MULTI-MNIST':
            torch.save(model.state_dict(), 'weights/Multi-MNIST_ae.pkl')
        if dataset_name == 'MULTI-USPS':
            torch.save(model.state_dict(), 'weights/Multi-MNIST-USPS_ae.pkl')
        if dataset_name == 'MULTI-FASHION':
            torch.save(model.state_dict(), 'weights/Multi-FASHION_ae.pkl')
        if dataset_name == 'MULTI-MVP-N':
            torch.save(model.state_dict(), 'weights/Multi-MVP-N_ae.pkl') 
        if dataset_name == 'MULTI_STL-10':
            torch.save(model.state_dict(), 'weights/Multi-STL-10_ae.pkl') 
        if dataset_name == 'MULTI_COIL_10':
            torch.save(model.state_dict(), 'weights/MULTI_COIL_10_ae.pkl')             
        if dataset_name == 'MULTI_COIL_20':
            torch.save(model.state_dict(), 'weights/MULTI_COIL_20_ae.pkl')  
        if dataset_name == 'MULTI_KITTI_00':
            torch.save(model.state_dict(), 'weights/MULTI_KITTI_00_ae.pkl') 
        if dataset_name == 'MULTI-CIFAR-10':
            torch.save(model.state_dict(), 'weights/MULTI_CIFAR_10_ae.pkl') 
        if dataset_name == 'MULTI-CelebA':
            torch.save(model.state_dict(), 'weights/MULTI-CelebA_ae.pkl') 
        if dataset_name == 'MULTI_Eglin':
            torch.save(model.state_dict(), 'weights/MULTI_Eglin_ae.pkl') 
                        
        print("model saved to weights/'dataset_name'_ae.pkl")

    else:
        if dataset_name == 'Dataset: Multi-MNIST':  
            load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/MNIST_ae.pkl'    
            model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'Dataset: Multi-FASHION': 
            load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/FASHION_ae.pkl'    
            model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'Fashion':  
            load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/Fashion_ae.pkl'    
            model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'Dataset: Multi-Market':   
            load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/Market_ae.pkl'    
            model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'Dataset: Multi-MVP-N':  
            load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/MVP-N_ae.pkl'    
            model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'Dataset: Multi-STL-10':   
            load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/STL-10_ae.pkl'    
            model.load_state_dict(torch.load(load_model_path))         
        if dataset_name == 'MULTI-MNIST':   
             load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/Multi-MNIST_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path))     
        if dataset_name == 'MULTI-USPS':   
             load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/Multi-MNIST-USPS_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path))   
        if dataset_name == 'MULTI-FASHION':   
             load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/Multi-FASHION_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'MULTI-MVP-N':   
             load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/Multi-MVP-N_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'MULTI_STL-10':  
             load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/Multi-STL-10_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'MULTI_COIL_10':
             load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/MULTI_COIL_10_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'MULTI_COIL_20':   
             load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/MULTI_COIL_20_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'MULTI_KITTI_00':   
             load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/MULTI_KITTI_00_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path))              
        if dataset_name == 'MULTI-CIFAR-10':   
             load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/MULTI_CIFAR_10_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path))  
        if dataset_name == 'MULTI-CelebA':   
             load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/MULTI-CelebA_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path))  
        if dataset_name == 'MULTI_Eglin':   
             load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/A_2) CCU-DEC/weights/MULTI_Eglin_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path))  
        
    # --- Grab one batch and visualize recon for a chosen view
    SHOW_VIEW = 0  # 0 = view 1
    for batch_idx, batch in enumerate(data_loader):
        xs, labels, _ = unpack_views(batch, V)
        with torch.no_grad():
            x = xs[SHOW_VIEW].to(device)
            x_hat, latent_c, latent_u = model(x)   # model is AE here (self.ae)
        break
    # --- Plot the original and reconstructed images
    Show_Complete_Reconstructions(x, x_hat)
    
    # optional cleanup
    x = x_hat = xs = batch = None
    torch.cuda.empty_cache()
#----------------------------------------------------------------------------

#*****************************************************************************
#--- Main Function
#*****************************************************************************
if __name__=='__main__': 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
      
    print('Loading data...')
    if dataset_name == 'MULTI-MNIST':
        dataset, dims, view, data_size, class_num = load_data("MULTI-MNIST")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI-USPS':
        dataset, dims, view, data_size, class_num = load_data("MULTI-USPS")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI-FASHION':
        dataset, dims, view, data_size, class_num = load_data("MULTI-FASHION")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'Fashion':
        dataset, dims, view, data_size, class_num = load_data("Fashion")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI_COIL_10':
        dataset, dims, view, data_size, class_num = load_data("MULTI_COIL_10")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI_COIL_20':
        dataset, dims, view, data_size, class_num = load_data("MULTI_COIL_20")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI-MVP-N':
        dataset, dims, view, data_size, class_num = load_data("MULTI-MVP-N")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI_STL-10':
        dataset, dims, view, data_size, class_num = load_data("MULTI-STL-10")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI-CIFAR-10':
        dataset, dims, view, data_size, class_num = load_data("MULTI-CIFAR-10")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI-CelebA':
        dataset, dims, view, data_size, class_num = load_data("MULTI-CelebA")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI_Eglin':
        dataset, dims, view, data_size, class_num = load_data("MULTI_Eglin")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI_KITTI_00':
        train_loader, test_loader, _, _ = get_KITTI_Multi_View_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)


    #--- Show all the Settings
    Show_settings() 
     
    #--- Show the dataloader images (V views)
    for _, batch in tqdm(enumerate(test_loader),
            total=int(len(test_loader.dataset) / test_loader.batch_size)
        ):
        xs, labels, _ = unpack_views(batch, V)   # xs is list length V
        break
    # Show view 0 against each other view
    for v in range(1, V):
        Show_dataloader_data(xs[0], xs[v], labels)
    
    # Clear references
    xs = None
    batch = None
    torch.cuda.empty_cache()
    
    #--- Define the Common and Peculiar Model
    model = C_U_Model(LATENT_DIM_C, LATENT_DIM_U, backbone=BACKBONE, img_size=IMG_SIZE).to(device)

    #--- Train the model with Common loss
    model.pretrain(test_loader)             

    #--- Create optimizer
    optimizer1 = Adam(model.parameters(), LR)
  
    # --- Initialize cluster centers (streaming, no full-dataset CUDA tensor)
    c, y_true = compute_common_features_stream(model, test_loader, V, device)  # both on CPU    
    
    # --- Cluster based on the (multi-view) common representation
    kmeans = KMeans(n_clusters=NUM_CLASSES, n_init=50)
    y_pred = kmeans.fit_predict(c.cpu().numpy())
    nmi_k = nmi_score(y_pred, y_true)
    print("Start Kmeans nmi score={:.4f}".format(nmi_k))
    
    # Option 1 (simple): use KMeans centroids directly as cluster_layer
    cluster_centers = torch.tensor(kmeans.cluster_centers_, 
                                   device=device, 
                                   dtype=c.dtype)
    print("Cluster centers: ", cluster_centers.size())
    #model.cluster_layer.data = cluster_centers   
    model.cluster_layer.data.copy_(cluster_centers)

    #--- Show the latent space
    c_plot, u_plot, y_plot = collect_plot_embeddings_stream(model, test_loader, V, device, view_idx=0, max_samples=5000)
    
    Show_Embedding_Space(c_plot, u_plot, y_plot)
    
    #--- Soft labels only based off the common representation 
    y_true_mv, y_pred_mv = eval_multiview_model(model, test_loader, V, device)
    acc = cluster_acc(y_true_mv, y_pred_mv, NUM_CLASSES)
    nmi = nmi_score(y_true_mv, y_pred_mv)
    pur = calculate_purity(y_true_mv, y_pred_mv)
    print('Acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', purity {:.4f}'.format(pur))
    #---------------------------------------------------

    #--- Show the cluster centers "Common" representation    
    centers = model.cluster_layer.data
    #--- Semantic labels
    center_labels = torch.arange(NUM_CLASSES).unsqueeze(1)
    zeros = torch.zeros(NUM_CLASSES, LATENT_DIM_U).to(device)
    j = torch.cat([centers, zeros], dim=1)
    centers = j.to(device) 
    with torch.no_grad():
        c = model.ae.decoder(centers)          
    Show_Representation(c, center_labels) 
    
    Start_ACC = acc
    Start_NMI = nmi
    Start_PUR = pur
    
    # Clear all unneeded data
    data_a = data_p = data_p2 = y_true = cluster_est = None
    torch.cuda.empty_cache()
    
    #**************************************************************************************
    #--- Now run stage II clustering loss
    #**************************************************************************************
    model.train()
    loss_fn = Cluster_loss_SoftPullAssigned(gamma=GAMMA, w_pull=W_PULL).to(device)
    
    mu_est = [torch.zeros(data_size, LATENT_DIM_C, device=device),
              torch.zeros(data_size, device=device, dtype=torch.long)]
        
    for epoch in range(FINE_TUNE_EPOCHS):
        total_loss = 0.0
    
        if epoch % UPDATE_INTERVAL == 0:
            with torch.no_grad():
                mu_est = [torch.zeros(data_size, LATENT_DIM_C, device=device),
                          torch.zeros(data_size, device=device, dtype=torch.long)]
                y_pred_total, y_true_list = [], []
    
                for batch in test_loader:
                    xs, y, idx = unpack_views(batch, V)
                    xs = [x.to(device) for x in xs]
    
                    cluster_est = model(*xs)
    
                    mu_est[0][idx] = model.cluster_layer.data[cluster_est].detach()
                    mu_est[1][idx] = idx.to(device).long()
    
                    y_pred_total.extend(cluster_est.cpu().numpy())
                    y_true_list.extend(y.cpu().numpy())
    
                acc = cluster_acc2(y_true_list, y_pred_total)
                nmi = nmi_score(y_true_list, y_pred_total)
                pur = calculate_purity(y_true_list, y_pred_total)
                print(f'Iter {epoch}: Acc {acc:.4f}, nmi {nmi:.4f}, purity {pur:.4f}')
    
        for i, batch in enumerate(test_loader):
            xs, y_true, idx = unpack_views(batch, V)
            xs = [x.to(device) for x in xs]
            idx = idx.to(device)
    
            optimizer1.zero_grad()
    
            xhats, cs = [], []
            for x in xs:
                x_hat, c, _ = model.ae(x)
                xhats.append(x_hat)
                cs.append(c)

            loss = loss_fn(mu_est[0][idx], cs, xhats, xs)
    
            total_loss += loss.item()
            loss.backward()
            optimizer1.step()
    
        print(f"epoch {epoch} loss={total_loss / (i + 1):.4f}")  
        
    # Clear unused data
    torch.cuda.empty_cache()        
        
    #--- Estimate Ending accuracy 
    # -------------------------
    # Collect full dataset views
    # -------------------------
    views_buf = [[] for _ in range(V)]
    y_true_buf = []
    
    for batch in test_loader:
        xs, y, idx = unpack_views(batch, V)
        for v in range(V):
            views_buf[v].append(xs[v])
        y_true_buf.append(y)
    
    views_all = [torch.cat(views_buf[v]).to(device) for v in range(V)]  # list of [N,C,H,W]
    y_true = torch.cat(y_true_buf)                                      # [N]
    
    # Clear CPU buffers
    views_buf = None
    y_true_buf = None
    torch.cuda.empty_cache()
    
    # ---------------------------------------
    # Encode commons (and one u for plotting)
    # ---------------------------------------
    with torch.no_grad():
        c_list = []
        for v in range(V):
            _, c_v, _ = model.ae(views_all[v])
            c_list.append(c_v)
    
        # For visualization: mimic your original Show_Embedding_Space(c_a, u, y_true)
        # Use view 0's c and u (same spirit as your original c_a/u)
        _, c_plot, u_plot = model.ae(views_all[0])
    
    Show_Embedding_Space(c_plot, u_plot, y_true)
    
    # ---------------------------------------
    # Measure view-invariance (avg pairwise)
    # ---------------------------------------
    c_list_n = [F.normalize(c_v, dim=1) for c_v in c_list]
    
    pair_dists = []
    for i in range(V):
        for j in range(i + 1, V):
            pair_dists.append(torch.norm(c_list_n[i] - c_list_n[j], dim=1).mean())
    
    avg_dist = torch.stack(pair_dists).mean().item() if len(pair_dists) else 0.0
    print(f"Average view-invariance dist (L2): {avg_dist:.4f}")
    
    # ---------------------------------------
    # Cluster using the model (option noise)
    # ---------------------------------------
    # Stage II metrics 
    y_true_mv, y_pred_mv = eval_multiview_model(model, test_loader, V, device)
    end_acc = cluster_acc2(y_true_mv, y_pred_mv)
    end_nmi = nmi_score(y_true_mv, y_pred_mv)
    end_pur = calculate_purity(y_true_mv, y_pred_mv)
    print(f'Acc {end_acc:.4f}, nmi {end_nmi:.4f}, purity {end_pur:.4f}')
    
    END_ACC = end_acc
    END_NMI = end_nmi
    END_PUR = end_pur
    Show_Results(SEED, Start_ACC, Start_NMI, Start_PUR, END_ACC, END_NMI, END_PUR)    
    
