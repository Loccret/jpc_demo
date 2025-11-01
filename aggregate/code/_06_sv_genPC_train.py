
import jpc

import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import optax

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from _01_utilities import get_mnist_loaders, plot_mnist_imgs
import warnings


def evaluate(key, layer_sizes, batch_size, network, test_loader, max_t1):
    test_acc = 0
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        acc, img_preds = jpc.test_generative_pc(
            model=network,
            input=label_batch,
            output=img_batch,
            key=key,
            layer_sizes=layer_sizes,
            batch_size=batch_size,
            max_t1=max_t1
        )
        test_acc += acc

    avg_test_acc = test_acc / len(test_loader)

    return avg_test_acc, label_batch, img_preds


def train_sv_gen_pc(
    seed,
    input_dim,
    width,
    depth,
    output_dim,
    act_fn,
    batch_size,
    lr,
    max_t1,
    test_every,
    n_train_iters,
    writter = None
):
    key = jax.random.PRNGKey(seed)
    key, *subkeys = jax.random.split(key, 4)
    network = jpc.make_mlp(
        key,
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=output_dim,
        act_fn=act_fn,
        use_bias=True
    )
    layer_sizes = [input_dim] + [width]*(depth-1) + [output_dim]
    optim = optax.adam(lr)
    opt_state = optim.init(
        (eqx.filter(network, eqx.is_array), None)
    )
    train_loader, test_loader = get_mnist_loaders(batch_size)

    for iter, (img_batch, label_batch) in tqdm(enumerate(train_loader), total = len(train_loader)):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        result = jpc.make_pc_step(
            model=network,
            optim=optim,
            opt_state=opt_state,
            input=label_batch,
            output=img_batch,
            max_t1=max_t1
        )
        network, opt_state = result["model"], result["opt_state"]
        train_loss = result["loss"]
        if ((iter+1) % test_every) == 0 or (iter+1) == n_train_iters:
            avg_test_acc, test_label_batch, img_preds = evaluate(
                key,
                layer_sizes,
                batch_size,
                network,
                test_loader,
                max_t1=max_t1
            )
            print(
                f"Train iter {iter+1}, train loss={train_loss:4f}, "
                f"avg test accuracy={avg_test_acc:4f}"
            )
            if (iter+1) >= n_train_iters:
                break

    fig = plot_mnist_imgs(img_preds, label_batch)
    # log to tensorboard
    if writter is not None:
        writter.add_scalar("SupervisedPC/test/accuracy", float(avg_test_acc), iter+1)
        writter.add_figure(
            tag="SupervisedPC/GeneratedImages",
            figure=fig,
            global_step=iter+1
        )
    plt.close(fig)
    return network
