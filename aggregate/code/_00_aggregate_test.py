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
from _01_utilities import get_mnist_loaders, get_cifar10_loaders, get_speechcommands_loaders
from _02_BiPC_train import train_BiPC
from _03_HPC_train import train_HPC
from _04_DPC_train import train_DPC
from _05_muPC_train import train_muPC
from aggregate.code._06_sv_genPC_train import train_sv_gen_pc
from typing import Tuple, Dict
import argparse

import jpc



def get_dataset_config(dataset_type: str) -> Dict:
    configs = {
        "MNIST": {
            'batch_size': 64
        },
        "CIFAR10": {
            'batch_size': 64
        },
        "SPEECHCOMMANDS": {
            'batch_size': 32  # Smaller batch size for audio data
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
    elif dataset_type == "CIFAR10":
        return get_cifar10_loaders(
            **get_dataset_config(dataset_type)
        )
    elif dataset_type == "SPEECHCOMMANDS":
        return get_speechcommands_loaders(
            **get_dataset_config(dataset_type)
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def get_data_dimensions(dataset_type: str) -> Tuple[int, int]:
    """Get input and output dimensions based on dataset type."""
    data_dims = {
        "MNIST": (784, 10),  # 28*28 flattened input, 10 classes
        "CIFAR10": (3072, 10),  # 32*32*3 flattened input, 10 classes  
        "SPEECHCOMMANDS": (16000, 12)  # 16000 audio samples, 12 speech command classes
    }
    if dataset_type not in data_dims:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return data_dims[dataset_type]


def get_model_config(model_type: str, dataset_type: str = "MNIST", 
                    shared_params: Dict = None) -> Dict:
    """
    Get model configuration with data-driven dimensions and shared hyperparameters.
    
    Args:
        model_type: Type of model ("BiPC", "HPC", "DPC", "muPC", "sv_gen_pc")
        dataset_type: Type of dataset ("MNIST", "CIFAR10", "SPEECHCOMMANDS") 
        shared_params: Dictionary of shared hyperparameters to override defaults
    """
    data_dim, label_dim = get_data_dimensions(dataset_type)
    
    # Default shared hyperparameters
    default_shared = {
        'seed': 0,
        'act_fn': "relu",
        'width': 300,
        'depth': 3,
        'test_every': 200,
        'n_train_iters': 300
    }
    
    # Update with user-provided shared parameters
    if shared_params:
        default_shared.update(shared_params)
    
    # Scale solver parameters based on dataset size for continuous-time models
    # Larger datasets need more solver steps and longer integration times
    # MNIST: 784 dims → scale=1.0, CIFAR10: 3072 dims → scale=3.9, SPEECHCOMMANDS: 16000 dims → scale=20.4
    solver_scale = max(1.0, data_dim / 784)
    
    # Calculate max_steps for solver - larger datasets need more steps
    default_max_steps = int(16384 * solver_scale)  # Scale from default 16384
    
    # Model-specific configurations
    model_specifics = {
        "BiPC": {
            'input_dim': label_dim,  # latent dimension for generative model
            'output_dim': data_dim,  # generate data
            'activity_lr': 5e-1,
            'param_lr': 1e-3,
            'test_every': 100,
        },
        "HPC": {
            'input_dim': label_dim,  # latent dimension for generative model
            'output_dim': data_dim,  # generate data
            'lr': 1e-3,
            'max_t1': int(50 * solver_scale),  # Scale integration time with data size
            'max_steps': default_max_steps,  # Scale max solver steps with data size
            'test_every': 1000,
        },
        "DPC": {
            'input_dim': data_dim,  # data input
            'output_dim': label_dim,  # classify data
            'use_bias': True,
            'lr': 1e-3,
        },
        "muPC": {
            'seed': 4329,  # specific seed for reproducibility
            'input_dim': data_dim,  # data input
            'output_dim': label_dim,  # classify data
            'width': 128,  # different width for muPC
            'depth': 30,   # much deeper for muPC
            'param_type': "mupc",
            'activity_lr': 5e-1,
            'param_lr': 1e-1,
            'n_train_iters': 900,  # more training iterations
        },
        "sv_gen_pc": {
            'input_dim': label_dim,  # latent dimension for generative model
            'output_dim': data_dim,  # generate data
            'lr': 1e-3,
            'max_t1': int(100 * solver_scale),  # Scale integration time with data size
            'max_steps': default_max_steps,  # Scale max solver steps with data size
            'n_train_iters': 200,  # fewer training iterations
        }
    }
    
    if model_type not in model_specifics:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Combine shared and model-specific parameters
    config = default_shared.copy()
    config.update(model_specifics[model_type])
    
    # Log solver scaling for continuous-time models
    if model_type in ["HPC", "sv_gen_pc"]:
        if "max_t1" in config and "max_steps" in config:
            print(f"  {model_type} max_t1: {config['max_t1']}, max_steps: {config['max_steps']} (solver_scale: {solver_scale:.2f})")
        elif "max_t1" in config:
            print(f"  {model_type} max_t1: {config['max_t1']} (solver_scale: {solver_scale:.2f})")
    
    return config


def create_writer(log_dir):
    if type(log_dir) is str:
        log_dir = Path(log_dir)

    paths = sorted(list(log_dir.glob('*/')))
    if len(paths) == 0:
        return SummaryWriter(log_dir=log_dir / 'run_000')
    last_index = int(paths[-1].name.split('_')[-1])
    new_index = last_index + 1
    return SummaryWriter(log_dir=log_dir / f'run_{new_index:03d}')

def run_pipeline(dataset_type: str, shared_hyperparams: Dict = None):
    """
    Run the complete training pipeline for all PC variants.
    
    Args:
        dataset_type: Dataset to use ("MNIST", "CIFAR10", "SPEECHCOMMANDS")
        shared_hyperparams: Dictionary of shared hyperparameters to apply to all models
    """
    writter = create_writer(log_dir=f'logs/{dataset_type}')
    
    # Get data loaders for the specified dataset
    train_loader, test_loader = load_dataset(dataset_type)
    
    # print("Training BiPC Model (generative PC...)")
    # train_BiPC(
    #     **get_model_config("BiPC", dataset_type, shared_hyperparams),
    #     batch_size=get_dataset_config(dataset_type)['batch_size'],
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     writter=writter,
    #     dataset_type=dataset_type
    # )

    # print("Training HPC Model (generative PC...)")
    # train_HPC(
    #     **get_model_config("HPC", dataset_type, shared_hyperparams),
    #     batch_size=get_dataset_config(dataset_type)['batch_size'],
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     writter=writter,
    #     dataset_type=dataset_type
    # )

    print("Training sv_gen_pc Model (generative and discriminative PC...)")
    train_sv_gen_pc(
        **get_model_config("sv_gen_pc", dataset_type, shared_hyperparams),
        batch_size=get_dataset_config(dataset_type)['batch_size'],
        train_loader=train_loader,
        test_loader=test_loader,
        writter=writter,
        dataset_type=dataset_type
    )

    print("Training DPC Model (discriminative PC...)")
    train_DPC(
        **get_model_config("DPC", dataset_type, shared_hyperparams),
        batch_size=get_dataset_config(dataset_type)['batch_size'],
        train_loader=train_loader,
        test_loader=test_loader,
        writer=writter
    )

    print("Training muPC Model")
    train_muPC(
        **get_model_config("muPC", dataset_type, shared_hyperparams),
        batch_size=get_dataset_config(dataset_type)['batch_size'],
        train_loader=train_loader,
        test_loader=test_loader,
        writer=writter
    )


def main():
    parser = argparse.ArgumentParser(description="Train Predictive Coding Variants")
    parser.add_argument('--dataset', type=str, default='CIFAR10', 
                       help='Dataset to use: MNIST, CIFAR10, or SPEECHCOMMANDS (default: MNIST)')
    parser.add_argument('--width', type=int, default=None,
                       help='Shared network width for all models')
    parser.add_argument('--depth', type=int, default=None,
                       help='Shared network depth for all models')
    parser.add_argument('--n_train_iters', type=int, default=None,
                       help='Shared number of training iterations for all models')
    parser.add_argument('--test_every', type=int, default=None,
                       help='Shared test frequency for all models')
    parser.add_argument('--seed', type=int, default=None,
                       help='Shared random seed for all models')
    parser.add_argument('--max_t1', type=int, default=None,
                       help='Max integration time for continuous-time models (HPC, sv_gen_pc)')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Max solver steps for continuous-time models (HPC, sv_gen_pc)')
    
    args = parser.parse_args()
    
    # Build shared hyperparameters dictionary from command line args
    shared_params = {}
    for param in ['width', 'depth', 'n_train_iters', 'test_every', 'seed', 'max_t1', 'max_steps']:
        value = getattr(args, param)
        if value is not None:
            shared_params[param] = value
    
    # Print configuration
    print(f"Running pipeline with dataset: {args.dataset}")
    if shared_params:
        print(f"Shared hyperparameters: {shared_params}")
    
    run_pipeline(dataset_type=args.dataset, shared_hyperparams=shared_params)


# # Example of how to use with custom shared parameters programmatically
# def example_custom_training():
#     """Example showing how to customize shared hyperparameters programmatically."""
#     # Custom shared hyperparameters
#     custom_shared = {
#         'width': 512,      # Wider networks
#         'depth': 4,        # Deeper networks
#         'n_train_iters': 500,  # More training
#         'seed': 42         # Reproducible results
#     }
    
#     print("Running custom training with shared hyperparameters:")
#     print(f"Shared params: {custom_shared}")
    
#     # Run with custom shared parameters
#     run_pipeline(dataset_type="MNIST", shared_hyperparams=custom_shared)

if __name__ == "__main__":
    main()