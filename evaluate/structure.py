import jpc

import jax.random as jr
import equinox as eqx
import equinox.nn as nn
import optax

import math
import random
import numpy as np
from typing import List, Callable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import warnings
warnings.simplefilter('ignore')  # ignore warnings


class ScaledLinear(eqx.Module):
    """Scaled linear transformation."""
    linear: nn.Linear
    scaling: float = eqx.static_field()
    
    def __init__(
            self,
            in_features,
            out_features,
            *,
            key,
            scaling=1.,
            param_type="sp",
            use_bias=False
    ):
        keys = jr.split(key, 2)
        linear = nn.Linear(
            in_features, 
            out_features, 
            use_bias=use_bias,
            key=keys[0]
        )
        if param_type == "mupc":
            W = jr.normal(keys[1], linear.weight.shape)
            linear = eqx.tree_at(lambda l: l.weight, linear, W)

        self.linear = linear
        self.scaling = scaling

    def __call__(self, x):
        return self.scaling * self.linear(x)
        

class ResNetBlock(eqx.Module):
    """Identity residual block applying activation and a scaled linear layer."""
    act_fn: Callable = eqx.static_field()
    scaled_linear: ScaledLinear

    def __init__(
        self,
        in_features,
        out_features,
        *,
        key,
        scaling=1.,
        param_type="sp",
        use_bias=False,
        act_fn="linear"
    ):
        self.act_fn = act_fn
        self.scaled_linear = ScaledLinear(
            in_features=in_features,
            out_features=out_features,
            key=key,
            scaling=scaling,
            param_type=param_type,
            use_bias=use_bias
        )
    # print("skip link is banned")
    print("skip link is activated")
    def __call__(self, x):
        res_path = x
        x = self.act_fn(x)
        return self.scaled_linear(x) + res_path


class Readout(eqx.Module):
    """Final network layer applying activation and a scaled linear layer."""
    act_fn: Callable = eqx.static_field()
    scaled_linear: ScaledLinear

    def __init__(
        self,
        in_features,
        out_features,
        *,
        key,
        scaling=1.,
        param_type="sp",
        use_bias=False,
        act_fn="linear"
    ):
        self.act_fn = act_fn
        self.scaled_linear = ScaledLinear(
            in_features=in_features,
            out_features=out_features,
            key=key,
            scaling=scaling,
            param_type=param_type,
            use_bias=use_bias
        )

    def __call__(self, x):
        x = self.act_fn(x)
        return self.scaled_linear(x)


class FCResNet(eqx.Module):
    """Fully-connected ResNet compatible with different parameterisations."""
    layers: List[eqx.Module]
    
    def __init__(
            self, 
            *,
            key, 
            in_dim, 
            width, 
            depth, 
            out_dim, 
            act_fn="linear", 
            use_bias=False,
            param_type="sp"
        ):
        act_fn = jpc.get_act_fn(act_fn)
        if param_type == "sp":
            in_scaling = 1.
            hidden_scaling = 1.
            out_scaling = 1.
        
        elif param_type == "mupc":
            in_scaling = 1 / math.sqrt(in_dim)
            hidden_scaling = 1 / math.sqrt(width * depth)
            out_scaling = 1 / width
            
        keys = jr.split(key, depth)
        self.layers = [
            ScaledLinear(
                key=keys[0],
                in_features=in_dim,
                out_features=width,
                scaling=in_scaling,
                param_type=param_type,
                use_bias=use_bias
            )
        ]

        for i in range(1, depth - 1):
            self.layers.append(
                ResNetBlock(
                    key=keys[i],
                    in_features=width,
                    out_features=width,
                    scaling=hidden_scaling,
                    param_type=param_type,
                    use_bias=use_bias,
                    act_fn=act_fn
                )
            )

        self.layers.append(
            Readout(
                key=keys[-1],
                in_features=width,
                out_features=out_dim,
                scaling=out_scaling,
                param_type=param_type,
                use_bias=use_bias,
                act_fn=act_fn
            )
        )

    def __call__(self, x):
        for f in self.layers:
            x = f(x)      
        return x

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]
