"""
Drafted by Juntang Wang at Mar 5th 4 the GASNN project

This file contains utility functions for the project.
"""
import platform
import sys
from typing import Tuple
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, Dataset
import matplotlib.pyplot as plt
import scienceplots


def print_environment_info():
    """Print information about the environment."""
    print("\n=== Environment Information ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Platform: {platform.platform()}")
    print("============================\n")


def get_device_info():
    """Get information about available computing devices.

    Returns
    -------
    torch.device
        The device that will be used for computations (either 'cuda' or 'cpu')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== Device Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Using device: {device}")
    print("========================\n")
    return device


def set_all_seeds(seed=42):
    """
    Set all seeds for reproducibility.
    """
    print("\n=== Setting Random Seeds ===")
    print(f"Seed value: {seed}")
    print("Setting torch CPU seed...")
    torch.manual_seed(seed)
    print("Setting torch CUDA seed...")
    torch.cuda.manual_seed_all(seed)
    print("Setting numpy seed...")
    np.random.seed(seed)
    print("Configuring CUDNN...")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Configuring PL...")
    pl.seed_everything(seed, workers=True)
    print("========================\n")
    return seed


def set_plt_style():
    """
    Set the style for matplotlib.
    """
    plt.style.use('science')
    plt.rcParams.update({
        'pdf.fonttype': 42,            # Use TrueType fonts in PDF 
        'ps.fonttype': 42,             # Use TrueType fonts in PS files
        'font.family': 'sans-serif',
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'figure.figsize': (10, 7),
        'font.size': 13,
        'axes.labelsize': 17,
        'axes.titlesize': 17,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        # 'text.usetex': True,
    })


def train_val_split(train_set, val_ratio=0.2, seed=42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and validation sets.

    Parameters
    ----------
    train_set : Dataset
        The dataset to be split.
    val_ratio : float, optional
        The ratio of the dataset to be used for validation (default is 0.2).
    seed : int, optional
        The seed for random number generation (default is 42).

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training and validation datasets.
    """
    train_set_size = int(len(train_set) * (1 - val_ratio))
    val_set_size = len(train_set) - train_set_size
    seed = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        train_set, [train_set_size, val_set_size], generator=seed
    )
    return train_set, val_set

