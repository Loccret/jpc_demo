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


def plot_mnist_imgs(imgs, labels, n_imgs=16):

    rows = np.sqrt(n_imgs).astype(int)
    cols = int(np.ceil(n_imgs / rows))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axs = axs.flatten()
    for i in range(n_imgs):
        ax = axs[i]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.imshow(imgs[i].reshape(28, 28), cmap=plt.cm.binary_r)
        ax.set_xlabel(jnp.argmax(labels, axis=1)[i])
    return fig