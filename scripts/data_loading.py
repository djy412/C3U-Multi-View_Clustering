# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:18:02 2026
@author: djy41
"""
from pathlib import Path
import os

import imageio.v2 as imageio  # pip install imageio
import numpy as np
import pandas as pd
import scipy.io
import scipy.io as sio
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from scripts.config import (
    NORMALIZED,
    BATCH_SIZE,
    PRE_TRAIN_EPOCHS,
    MODEL_FILENAME,
    LR,
    WORKERS,
    LATENT_DIM_C,
    LATENT_DIM_U,
    NUM_CLASSES,
    LAMBDA,
    GAMMA,
    FINE_TUNE_EPOCHS,
    dataset_name,
    CHANNELS,
    TOLERANCE,
    UPDATE_INTERVAL,
    DATA_ROOT,   # add this in scripts/config.py
)


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
DATA_ROOT = Path(DATA_ROOT)


def ds(*parts) -> Path:
    """Build a dataset path under the configured DATA_ROOT."""
    return DATA_ROOT.joinpath(*parts)


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def get_resized_transform():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform


def get_normalized_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform


def get_simple_transform():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform


# -----------------------------------------------------------------------------
# Generic CSV triplet dataset
# -----------------------------------------------------------------------------
class CSVTripletDataset(Dataset):
    """
    Generic dataset for CSV-backed triplets:
      col 0 -> anchor image path
      col 1 -> positive image path
      col 2 -> negative image path
      col 3 -> label
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.annotations = pd.read_csv(self.root_dir / csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        anchor = imread(self.root_dir / self.annotations.iloc[index, 0])
        positive = imread(self.root_dir / self.annotations.iloc[index, 1])
        negative = imread(self.root_dir / self.annotations.iloc[index, 2])
        y_label = torch.tensor(int(self.annotations.iloc[index, 3]))

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative, y_label, index


# -----------------------------------------------------------------------------
# CIFAR-10 seven-view CSV dataset
# -----------------------------------------------------------------------------
class CIFAR_Dataset(Dataset):
    """Dataset for CIFAR-10 'seven-views' CSV; returns V1, V2, V3, label, index."""
    def __init__(self, csv_file, root_dir="", transform=None):
        self.root_dir = Path(root_dir) if root_dir else Path()
        csv_path = Path(csv_file)
        if not csv_path.is_absolute():
            csv_path = self.root_dir / csv_path

        self.annotations = pd.read_csv(csv_path)
        self.transform = transform

        expected_cols = 8  # img1..img7 + class_id
        if self.annotations.shape[1] < expected_cols:
            raise ValueError(
                f"CSV has {self.annotations.shape[1]} columns; expected >= {expected_cols}."
            )

        for i in range(7):
            self.annotations.iloc[:, i] = self.annotations.iloc[:, i].astype(str)

    def __len__(self):
        return len(self.annotations)

    def _resolve_path(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.root_dir / p

    def __getitem__(self, index):
        row = self.annotations.iloc[index]

        p1 = self._resolve_path(row.iloc[0])
        p2 = self._resolve_path(row.iloc[1])
        p3 = self._resolve_path(row.iloc[2])

        V1 = imageio.imread(p1)
        V2 = imageio.imread(p2)
        V3 = imageio.imread(p3)

        if V1.ndim == 2:
            V1 = V1[..., None].repeat(3, axis=2)
        if V2.ndim == 2:
            V2 = V2[..., None].repeat(3, axis=2)
        if V3.ndim == 2:
            V3 = V3[..., None].repeat(3, axis=2)

        if self.transform:
            V1 = self.transform(V1)
            V2 = self.transform(V2)
            V3 = self.transform(V3)

        label = torch.tensor(int(row.iloc[7]), dtype=torch.long)
        return V1, V2, V3, label, index


def get_cifar10_data_loaders(batch_size=256, num_workers=10):
    transform = get_simple_transform()
    csv_file = "cifar10_seven_views.csv"
    root_dir = ds("cifar10_images")

    train_dataset = CIFAR_Dataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, train_dataset


# -----------------------------------------------------------------------------
# MVP-N
# -----------------------------------------------------------------------------
class MVPN_Dataset(CSVTripletDataset):
    """Dataset made with MVP-N 64x64 images"""
    pass


def get_MVPN_dataloaders(batch_size=256, num_workers=8):
    """MVP-N dataloader with (64x64) images."""
    transform = get_simple_transform()

    train_dataset = MVPN_Dataset(
        csv_file="data.csv",
        root_dir=ds("MVP-N_Triplet_Train"),
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )

    test_dataset = MVPN_Dataset(
        csv_file="data.csv",
        root_dir=ds("MVP-N_Triplet_Test"),
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2 * batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    return train_loader, test_loader, train_dataset, test_dataset


# -----------------------------------------------------------------------------
# Multi-Market
# -----------------------------------------------------------------------------
class Multi_Market_Dataset(CSVTripletDataset):
    """Dataset made with Multi-Market 64 x 128 images"""
    pass


def get_Multi_Market_dataloaders(batch_size=256, num_workers=8):
    """Multi_Market dataloader with (64x128) images."""
    transform = get_normalized_transform()

    train_dataset = Multi_Market_Dataset(
        csv_file="data.csv",
        root_dir=ds("Multi-Market_Triplet_Train"),
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )

    test_dataset = Multi_Market_Dataset(
        csv_file="data.csv",
        root_dir=ds("Multi-Market_Triplet_Test"),
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2 * batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    return train_loader, test_loader, train_dataset, test_dataset


# -----------------------------------------------------------------------------
# Multi-MNIST
# -----------------------------------------------------------------------------
class Multi_MNIST_Dataset(CSVTripletDataset):
    """Dataset made with Multi-MNIST 32 x 32 images"""
    pass


def get_Multi_MNIST_dataloaders(batch_size=256, num_workers=8):
    """Multi_MNIST dataloader with (32, 32) images."""
    transform = get_simple_transform()
    root_dir = ds("Multi_MNIST")

    train_dataset = Multi_MNIST_Dataset(
        csv_file="data.csv",
        root_dir=root_dir,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )

    test_dataset = Multi_MNIST_Dataset(
        csv_file="data.csv",
        root_dir=root_dir,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    return train_loader, test_loader, train_dataset, test_dataset


# -----------------------------------------------------------------------------
# Multi-FASHION
# -----------------------------------------------------------------------------
class Multi_FASHION_Dataset(CSVTripletDataset):
    """Dataset made with Multi-Fashion 32 x 32 images"""
    pass


def get_Multi_FASHION_dataloaders(batch_size=256, num_workers=8):
    """Multi_Fashion dataloader with (32, 32) images."""
    transform = get_simple_transform()
    root_dir = ds("Multi_Fashion_Test")

    train_dataset = Multi_FASHION_Dataset(
        csv_file="data.csv",
        root_dir=root_dir,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )

    test_dataset = Multi_FASHION_Dataset(
        csv_file="data.csv",
        root_dir=root_dir,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )

    return train_loader, test_loader, train_dataset, test_dataset


# -----------------------------------------------------------------------------
# Multi-STL-10
# -----------------------------------------------------------------------------
class Multi_STL_10_Dataset(CSVTripletDataset):
    """Dataset made with Multi-STL_10 64 x 64 images"""
    pass


def get_Multi_STL_10_dataloaders(batch_size=256, num_workers=8):
    """Multi_STL dataloader with (64, 64) images."""
    transform = get_simple_transform()
    root_dir = ds("Multi-STL_10_Train")

    train_dataset = Multi_STL_10_Dataset(
        csv_file="data.csv",
        root_dir=root_dir,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )

    test_dataset = Multi_STL_10_Dataset(
        csv_file="data.csv",
        root_dir=root_dir,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    return train_loader, test_loader, train_dataset, test_dataset


# -----------------------------------------------------------------------------
# KITTI multi-view
# -----------------------------------------------------------------------------
class KITTIMultiViewTriplets(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = Path(root_dir)
        self.annotations = pd.read_csv(self.root_dir / csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        img1_path = self.root_dir / row["img1"]
        img2_path = self.root_dir / row["img2"]
        img3_path = self.root_dir / row["img3"]
        y_label = torch.tensor(int(row["y"]), dtype=torch.long)

        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")
        img3 = Image.open(img3_path).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3, y_label, idx


def get_KITTI_Multi_View_dataloaders(batch_size=128, num_workers=8):
    """Multi Kitti dataloader with (128, 128) images."""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    root_dir = ds("Kitti", "Kitti_00_Multi_View")

    train_dataset = KITTIMultiViewTriplets(
        csv_file="data.csv",
        root_dir=root_dir,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )

    test_dataset = KITTIMultiViewTriplets(
        csv_file="data.csv",
        root_dir=root_dir,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )

    return train_loader, test_loader, train_dataset, test_dataset


# -----------------------------------------------------------------------------
# .mat datasets
# -----------------------------------------------------------------------------
class MULTI_USPS(Dataset):
    def __init__(self, path, filename="MNIST_USPS.mat", pad_to_32=True):
        mat_path = os.path.join(path, filename)
        mat = scipy.io.loadmat(mat_path)

        Y = mat["Y"].squeeze().astype(np.int64)
        V1 = mat["X1"].astype(np.float32)
        V2 = mat["X2"].astype(np.float32)

        self.Y = Y
        self.V1 = V1
        self.V2 = V2
        self.pad_to_32 = pad_to_32
        self.N = self.Y.shape[0]

    def __len__(self):
        return self.N

    def _to_chw(self, x):
        if x.ndim == 3:
            x = x.transpose(2, 0, 1)
        elif x.ndim == 2:
            x = x[None, ...]
        else:
            raise ValueError(f"Unexpected sample shape: {x.shape}")
        return x

    def __getitem__(self, idx):
        x1 = self._to_chw(self.V1[idx])
        x2 = self._to_chw(self.V2[idx])

        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)

        if self.pad_to_32:
            x1 = F.pad(x1, (2, 2, 2, 2))
            x2 = F.pad(x2, (2, 2, 2, 2))

        y = torch.tensor(self.Y[idx], dtype=torch.long)
        idx_t = torch.tensor(idx, dtype=torch.long)

        return x1, x2, y, idx_t


class MULTI_MNIST(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "MULTI_MNIST.mat")["Y"].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + "MULTI_MNIST.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "MULTI_MNIST.mat")["X2"].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(2, 0, 1)
        x2 = self.V2[idx].transpose(2, 0, 1)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


class MULTI_FASHION(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "MULTI_FASHION_Test.mat")["Y"].astype(np.int32).reshape(3333,)
        self.V1 = scipy.io.loadmat(path + "MULTI_FASHION_Test.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "MULTI_FASHION_Test.mat")["X2"].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + "MULTI_FASHION_Test.mat")["X3"].astype(np.float32)

    def __len__(self):
        return 3333

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(2, 0, 1)
        x2 = self.V2[idx].transpose(2, 0, 1)
        x3 = self.V3[idx].transpose(2, 0, 1)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "Fashion.mat")["Y"].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + "Fashion.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "Fashion.mat")["X2"].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + "Fashion.mat")["X3"].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(2, 0, 1)
        x2 = self.V2[idx].transpose(2, 0, 1)
        x3 = self.V3[idx].transpose(2, 0, 1)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


class MULTI_COIL_10(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "Multi-COIL-10.mat")["Y"].astype(np.int32).reshape(720,)
        self.V1 = scipy.io.loadmat(path + "Multi-COIL-10.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "Multi-COIL-10.mat")["X2"].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + "Multi-COIL-10.mat")["X3"].astype(np.float32)

    def __len__(self):
        return 720

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(0, 1, 2)
        x2 = self.V2[idx].transpose(0, 1, 2)
        x3 = self.V3[idx].transpose(0, 1, 2)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


class MULTI_COIL_20(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "MULTI_COIL-20.mat")["Y"].astype(np.int32).reshape(480,)
        self.V1 = scipy.io.loadmat(path + "MULTI_COIL-20.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "MULTI_COIL-20.mat")["X2"].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + "MULTI_COIL-20.mat")["X3"].astype(np.float32)

    def __len__(self):
        return 480

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(2, 0, 1)
        x2 = self.V2[idx].transpose(2, 0, 1)
        x3 = self.V3[idx].transpose(2, 0, 1)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


class MULTI_MVP_N(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "MULTI_MVP-N.mat")["Y"].astype(np.int32).reshape(3000,)
        self.V1 = scipy.io.loadmat(path + "MULTI_MVP-N.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "MULTI_MVP-N.mat")["X2"].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + "MULTI_MVP-N.mat")["X3"].astype(np.float32)

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(2, 0, 1)
        x2 = self.V2[idx].transpose(2, 0, 1)
        x3 = self.V3[idx].transpose(2, 0, 1)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


class MULTI_STL_10(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "MULTI_STL-10.mat")["Y"].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + "MULTI_STL-10.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "MULTI_STL-10.mat")["X2"].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + "MULTI_STL-10.mat")["X3"].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(2, 0, 1)
        x2 = self.V2[idx].transpose(2, 0, 1)
        x3 = self.V3[idx].transpose(2, 0, 1)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


class MULTI_STL_10_colorjitter(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "MULTI_STL-10_cj.mat")["Y"].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + "MULTI_STL-10_cj.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "MULTI_STL-10_cj.mat")["X2"].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + "MULTI_STL-10_cj.mat")["X3"].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(2, 0, 1)
        x2 = self.V2[idx].transpose(2, 0, 1)
        x3 = self.V3[idx].transpose(2, 0, 1)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


class MULTI_STL_10_RR(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "MULTI_STL-10_rr.mat")["Y"].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + "MULTI_STL-10_rr.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "MULTI_STL-10_rr.mat")["X2"].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + "MULTI_STL-10_rr.mat")["X3"].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(2, 0, 1)
        x2 = self.V2[idx].transpose(2, 0, 1)
        x3 = self.V3[idx].transpose(2, 0, 1)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


class MULTI_CIFAR_10(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "MULTI_CIFAR-10.mat")["Y"].astype(np.int32).reshape(20000,)
        self.V1 = scipy.io.loadmat(path + "MULTI_CIFAR-10.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "MULTI_CIFAR-10.mat")["X2"].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + "MULTI_CIFAR-10.mat")["X3"].astype(np.float32)

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(2, 0, 1)
        x2 = self.V2[idx].transpose(2, 0, 1)
        x3 = self.V3[idx].transpose(2, 0, 1)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


class MULTI_CelebA(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "MultiCelebA_Train.mat")["Y"].astype(np.int32).reshape(100,)
        self.V1 = scipy.io.loadmat(path + "MultiCelebA_Train.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "MultiCelebA_Train.mat")["X2"].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + "MultiCelebA_Train.mat")["X3"].astype(np.float32)

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(2, 0, 1)
        x2 = self.V2[idx].transpose(2, 0, 1)
        x3 = self.V3[idx].transpose(2, 0, 1)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


class MULTI_Eglin(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + "MULTI_Eglin.mat")["Y"].astype(np.int32).reshape(1000,)
        self.V1 = scipy.io.loadmat(path + "MULTI_Eglin.mat")["X1"].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + "MULTI_Eglin.mat")["X2"].astype(np.float32)

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        x1 = self.V1[idx].transpose(2, 0, 1)
        x2 = self.V2[idx].transpose(2, 0, 1)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            self.Y[idx],
            torch.from_numpy(np.array(idx)).long(),
        )


# -----------------------------------------------------------------------------
# Caltech101-7
# -----------------------------------------------------------------------------
class Caltech7Dataset(Dataset):
    def __init__(self, mat_path, views_to_use=(3, 4, 5), normalized=False):
        mat = sio.loadmat(mat_path)
        X = mat["X"]

        if X.ndim != 2:
            raise ValueError(f"Expected X to be 2D cell array, got shape {X.shape}")

        if X.shape[0] == 1:
            cells = [X[0, j] for j in range(X.shape[1])]
        elif X.shape[1] == 1:
            cells = [X[j, 0] for j in range(X.shape[0])]
        else:
            cells = list(X.ravel())

        self.views = [
            torch.from_numpy(np.asarray(cells[j], dtype=np.float32)).contiguous()
            for j in views_to_use
        ]

        if normalized:
            for k in range(len(self.views)):
                v = self.views[k]
                v_min = v.min(dim=0, keepdim=True).values
                v_max = v.max(dim=0, keepdim=True).values
                self.views[k] = (v - v_min) / (v_max - v_min + 1e-8)

        y = np.asarray(mat["Y"]).squeeze()
        self.labels = torch.from_numpy(y.astype(np.int64)) - 1

    def __len__(self):
        return self.views[0].shape[0]

    def __getitem__(self, idx):
        xs = [v[idx] for v in self.views]
        label = self.labels[idx]
        return (*xs, label, torch.tensor(idx, dtype=torch.long))


def get_Caltech101_7_dataloaders(batch_size, num_workers, path=None, normalized=False):
    if path is None:
        path = ds("Caltech101-7.mat")
    dataset = Caltech7Dataset(path, normalized=normalized)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return train_loader, test_loader, dataset, dataset


# -----------------------------------------------------------------------------
# Unified loader
# -----------------------------------------------------------------------------
def load_data(dataset):
    cwd = os.getcwd()

    if dataset == "MULTI-MNIST":
        dataset = MULTI_MNIST("./data/")
        dims = [1024, 1024]
        view = 2
        class_num = 10
        data_size = 5000

    elif dataset == "MULTI-USPS":
        dataset = MULTI_USPS("./data/")
        dims = [1024, 1024]
        view = 2
        class_num = 10
        data_size = len(dataset)

    elif dataset == "MULTI-FASHION":
        dataset = MULTI_FASHION("./data/")
        dims = [1024, 1024, 1024]
        view = 3
        class_num = 10
        data_size = 3333

    elif dataset == "Fashion":
        dataset = Fashion("./data/")
        dims = [1024, 1024, 1024]
        view = 3
        class_num = 10
        data_size = 10000

    elif dataset == "MULTI_COIL_10":
        dataset = MULTI_COIL_10("./data/")
        dims = [1024, 1024, 1024]
        view = 3
        class_num = 10
        data_size = 720

    elif dataset == "MULTI_COIL_20":
        dataset = MULTI_COIL_20("./data/")
        dims = [16384, 16384, 16384]
        view = 3
        class_num = 20
        data_size = 480

    elif dataset == "MULTI-MVP-N":
        dataset = MULTI_MVP_N("./data/")
        dims = [12288, 12288, 12288]
        view = 3
        class_num = 10
        data_size = 3000

    elif dataset == "MULTI_STL-10":
        dataset = MULTI_STL_10("./data/")
        dims = [27648, 27648, 27648]
        view = 3
        class_num = 10
        data_size = 5000

    elif dataset == "MULTI-CIFAR-10":
        dataset = MULTI_CIFAR_10("./data/")
        dims = [3072, 3072, 3072]
        view = 3
        class_num = 10
        data_size = 20000

    elif dataset == "MULTI-CelebA":
        dataset = MULTI_CelebA("./data/")
        dims = [27648, 27648, 27648]
        view = 3
        class_num = 10
        data_size = 100

    elif dataset == "MULTI_Eglin":
        dataset = MULTI_Eglin("./data/")
        dims = [27648, 27648]
        view = 2
        class_num = 4
        data_size = 1000

    elif dataset == "Caltech101_7":
        mat_path = ds("Caltech101-7.mat")
        dataset = Caltech7Dataset(mat_path, views_to_use=(1, 2), normalized=True)
        dims = [40, 254]
        view = 2
        class_num = 7
        data_size = len(dataset)

    else:
        print(dataset)
        raise NotImplementedError

    return dataset, dims, view, data_size, class_num