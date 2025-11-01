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
from _01_utilities import get_mnist_loaders
import warnings


def set_global_seed(seed):
    torch.manual_seed(seed)             
    torch.cuda.manual_seed(seed)            
    torch.cuda.manual_seed_all(seed)        
    np.random.seed(seed)                  
    random.seed(seed)                       
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 


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


def evaluate(model, skip_model, test_loader, param_type):
    avg_test_acc = 0
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        _, test_acc = jpc.test_discriminative_pc(
            model=model,
            input=img_batch,
            output=label_batch,
            skip_model=skip_model,
            param_type=param_type
        )
        avg_test_acc += test_acc

    return avg_test_acc / len(test_loader)

def train_muPC(
    seed,  
    input_dim,
    width,
    depth,
    output_dim,
    act_fn,
    param_type,
    activity_lr,  
    param_lr,
    batch_size,
    test_every,
    n_train_iters,
    writer = None
):  
    
    key = jr.PRNGKey(seed)
    model = jpc.make_mlp(
        key,
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=output_dim,
        act_fn=act_fn,
        param_type="mupc"
    )

    skip_model = jpc.make_skip_model(depth)
    #############################################
    # it is ok to remove this block
    mupc_model = FCResNet(
        key=key, 
        in_dim=input_dim, 
        width=width, 
        depth=depth, 
        out_dim=output_dim, 
        act_fn=act_fn, 
        use_bias=False, 
        param_type="mupc"
    )
    mupc_model = eqx.tree_at(
        where=lambda tree: tree[0].linear.weight,
        pytree=mupc_model,
        replace=model[0][1].weight
    )
    for l in range(1, len(model)):
        mupc_model = eqx.tree_at(
            where=lambda tree: tree[l].scaled_linear.linear.weight,
            pytree=mupc_model,
            replace=model[l][1].weight
        )
    ##########################################

    set_global_seed(seed)
    activity_optim = optax.sgd(activity_lr)
    param_optim = optax.adam(param_lr)
    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), skip_model)
    )
    train_loader, test_loader = get_mnist_loaders(batch_size)

    for iter, (img_batch, label_batch) in enumerate(train_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        # initialise activities
        activities = jpc.init_activities_with_ffwd(
            model=model,
            input=img_batch,
            skip_model=skip_model,
            param_type=param_type
        )
        activity_opt_state = activity_optim.init(activities)
        train_loss = jpc.mse_loss(activities[-1], label_batch)

        # inference
        for t in range(len(model)):
            activity_update_result = jpc.update_activities(
                params=(model, skip_model),
                activities=activities,
                optim=activity_optim,
                opt_state=activity_opt_state,
                output=label_batch,
                input=img_batch,
                param_type=param_type
            )
            activities = activity_update_result["activities"]
            activity_opt_state = activity_update_result["opt_state"]

        # learning
        param_update_result = jpc.update_params(
            params=(model, skip_model),
            activities=activities,
            optim=param_optim,
            opt_state=param_opt_state,
            output=label_batch,
            input=img_batch,
            param_type=param_type
        )
        model = param_update_result["model"]
        skip_model = param_update_result["skip_model"]
        param_opt_state = param_update_result["opt_state"]

        if np.isinf(train_loss) or np.isnan(train_loss):
            print(
                f"Stopping training because of divergence, train loss={train_loss}"
            )
            break
    
        if ((iter+1) % test_every) == 0 or (iter+1) == len(train_loader):
            avg_test_acc = evaluate(
                model=model,
                skip_model=skip_model, 
                test_loader=test_loader, 
                param_type=param_type
            )
            print(
                f"Train iter {iter+1}, train loss={train_loss:4f}, "
                f"avg test accuracy={avg_test_acc:4f}"
            )
            if writer is not None:
                writer.add_scalar("muPC/test/Accuracy", float(avg_test_acc), iter+1)
            if (iter+1) >= n_train_iters:
                break