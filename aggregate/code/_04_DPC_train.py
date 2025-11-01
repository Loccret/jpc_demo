import jpc

import jax
import equinox as eqx
import equinox.nn as nn
import optax

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.simplefilter('ignore')  # ignore warnings
from pathlib import Path
import numpy as np
import os
import pickle
from _01_utilities import get_mnist_loaders




def evaluate(model, test_loader):
    avg_test_loss, avg_test_acc = 0, 0
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
            
        test_loss, test_acc = jpc.test_discriminative_pc(
            model=model,
            input=img_batch,
            output=label_batch
        )
        avg_test_loss += test_loss
        avg_test_acc += test_acc

    return avg_test_loss / len(test_loader), avg_test_acc / len(test_loader)


def train_DPC(
    seed,
    input_dim,
    width,
    depth,
    output_dim,
    act_fn,
    use_bias,
    lr,
    batch_size,
    test_every,
    n_train_iters,
    writer = None,
    #   log_every = 10,
    #   save_gradients = True
):
    
    key = jax.random.PRNGKey(seed)
    model = jpc.make_mlp(
        key,
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=output_dim,
        act_fn=act_fn,
        use_bias=use_bias
    )
    optim = optax.adam(lr)
    opt_state = optim.init(
        (eqx.filter(model, eqx.is_array), None)
    )
    train_loader, test_loader = get_mnist_loaders(batch_size)


    CALCULATE_ACCURACY = True
    ACTIVITY_NORMS = True
    for iter, (img_batch, label_batch) in enumerate(train_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        result = jpc.make_pc_step(
            model=model,
            optim=optim,
            opt_state=opt_state,
            output=label_batch,
            input=img_batch,
            calculate_accuracy = CALCULATE_ACCURACY,
            activity_norms = ACTIVITY_NORMS
        )

        # if writer is not None and (iter % log_every) == 0:
        #     record_logs(writer, iter, result)

        model, opt_state = result["model"], result["opt_state"]
        train_loss = result["loss"]
        if ((iter+1) % test_every) == 0 or (iter+1) == len(train_loader):
            _, avg_test_acc = evaluate(model, test_loader)
            print(
                f"Train iter {iter+1}, train loss={train_loss:4f}, "
                f"avg test accuracy={avg_test_acc:4f}"
            )
            writer.add_scalar(
                "DPC/test/accuracy",
                float(avg_test_acc),
                iter+1
            )
            if (iter+1) >= n_train_iters:
                break

        

