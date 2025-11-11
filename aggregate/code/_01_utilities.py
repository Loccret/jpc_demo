import os
import pickle
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm



def one_hot(labels, n_classes=10):
    arr = torch.eye(n_classes)
    return arr[labels]


class MNIST(datasets.MNIST):
    def __init__(self, train, normalise=True, save_dir="data"):
        if normalise:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.1307), std=(0.3081)
                    )
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(save_dir, download=True, train=train, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = torch.flatten(img)
        label = one_hot(label)
        return img, label
    

class CIFAR10(datasets.CIFAR10):
    def __init__(self, train, normalise=True, save_dir="data"):
        if normalise:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010)
                    )
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(save_dir, download=True, train=train, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = torch.flatten(img)
        label = one_hot(label, n_classes=10)
        return img, label


def get_mnist_loaders(batch_size):
    train_data = MNIST(train=True, normalise=True)
    test_data = MNIST(train=False, normalise=True)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    return train_loader, test_loader


def get_cifar10_loaders(batch_size):
    train_data = CIFAR10(train=True, normalise=True)
    test_data = CIFAR10(train=False, normalise=True)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    return train_loader, test_loader

def plot_imgs(imgs, labels, n_imgs=16, dataset_type="MNIST"):
    """Plot images from different datasets."""
    rows = np.sqrt(n_imgs).astype(int)
    cols = int(np.ceil(n_imgs / rows))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axs = axs.flatten()
    
    for i in range(n_imgs):
        ax = axs[i]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        
        if dataset_type == "MNIST":
            # MNIST: 784 -> 28x28, grayscale
            ax.imshow(imgs[i].reshape(28, 28), cmap=plt.cm.binary_r)
        elif dataset_type == "CIFAR10":
            # CIFAR10: 3072 -> 32x32x3, color
            # Reshape and handle color channels
            img = imgs[i].reshape(32, 32, 3)
            # Clip values to valid range and convert to uint8 for display
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        else:
            # Fallback: try to detect format based on size
            if imgs[i].shape[0] == 784:
                ax.imshow(imgs[i].reshape(28, 28), cmap=plt.cm.binary_r)
            elif imgs[i].shape[0] == 3072:
                img = imgs[i].reshape(32, 32, 3)
                img = np.clip(img, 0, 1)
                ax.imshow(img)
            else:
                # Can't determine format, skip plotting
                ax.text(0.5, 0.5, 'Unknown\nformat', ha='center', va='center', 
                       transform=ax.transAxes)
        
        ax.set_xlabel(jnp.argmax(labels, axis=1)[i])
    return fig


def plot_mnist_imgs(imgs, labels, n_imgs=16):
    """Legacy function for backward compatibility - auto-detects dataset format."""
    # Auto-detect dataset type based on image dimensions
    if imgs[0].shape[0] == 784:
        dataset_type = "MNIST"
    elif imgs[0].shape[0] == 3072:
        dataset_type = "CIFAR10"
    else:
        dataset_type = "auto"  # Let plot_imgs handle unknown formats
    
    return plot_imgs(imgs, labels, n_imgs, dataset_type=dataset_type)