import os
import pickle
import warnings
from pathlib import Path

import equinox as eqx
import equinox.nn as nn
import jax
import numpy as np
import optax
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
from _01_utilities import get_mnist_loaders
from _02_BiPC_train import train_BiPC
from _03_HPC_train import train_HPC
from typing import Tuple, Dict
import argparse
import jpc



def get_dataset_config(dataset_type: str) -> Dict:
    configs = {
        "MNIST": {
            'batch_size': 64
        }
    }
    if dataset_type in configs:
        return configs[dataset_type]
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_dataset(dataset_type: str) -> Tuple[DataLoader, DataLoader]:
    if dataset_type == "MNIST":
        return get_mnist_loaders(
            **get_dataset_config(dataset_type)
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def get_model_config(dataset_type: str) -> Dict:
    configs = {
        "BiPC": {
            'seed': 0,
            'input_dim': 10,
            'width': 300,
            'depth': 3,
            'output_dim': 784,
            'act_fn': "relu",
            'activity_lr': 5e-1,
            'param_lr': 1e-3,
            'test_every': 100,
            'n_train_iters': 300
        },
        "HPC": {
            'seed': 0,
            'input_dim': 10,
            'width': 300,
            'depth': 3,
            'output_dim': 784,
            'act_fn': "relu",
            'lr': 1e-3,
            'max_t1': 50,
            'test_every': 100,
            'n_train_iters': 300
        }
    }
    if dataset_type in configs:
        return configs[dataset_type]
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def create_writer(log_dir):
    if type(log_dir) is str:
        log_dir = Path(log_dir)

    paths = sorted(list(log_dir.glob('*/')))
    if len(paths) == 0:
        return SummaryWriter(log_dir=log_dir / 'run_000')
    last_index = int(paths[-1].name.split('_')[-1])
    new_index = last_index + 1
    return SummaryWriter(log_dir=log_dir / f'run_{new_index:03d}')

def run_pipeline(dataset_type: str):
    writter = create_writer(log_dir=f'logs/{dataset_type}')

    # train_BiPC(
    #     **get_model_config("BiPC"),
    #     batch_size=get_dataset_config(dataset_type)['batch_size'],
    #     writter=writter
    # )
    train_HPC(
        **get_model_config("HPC"),
        batch_size=get_dataset_config(dataset_type)['batch_size'],
        writter=writter
    )




def main():
    parser = argparse.ArgumentParser(description="Train Predictive Coding Variants")
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use (default: MNIST)')
    args = parser.parse_args()
    run_pipeline(dataset_type=args.dataset)

if __name__ == "__main__":
    main()