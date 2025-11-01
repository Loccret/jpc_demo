import jpc

import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import optax

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from _01_utilities import get_mnist_loaders, plot_mnist_imgs
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')  # ignore warnings



def evaluate(
      key,
      layer_sizes,
      batch_size,
      generator,
      amortiser,
      test_loader
):
    amort_accs, hpc_accs, gen_accs = 0, 0, 0
    for _, (img_batch, label_batch) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        amort_acc, hpc_acc, gen_acc, img_preds = jpc.test_hpc(
            key=key,
            layer_sizes=layer_sizes,
            batch_size=batch_size,
            generator=generator,
            amortiser=amortiser,
            input=label_batch,
            output=img_batch
        )
        amort_accs += amort_acc
        hpc_accs += hpc_acc
        gen_accs += gen_acc

    return (
        amort_accs / len(test_loader),
        hpc_accs / len(test_loader),
        gen_accs / len(test_loader),
        label_batch,
        img_preds
    )


def train_HPC(
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
    key, *subkey = jax.random.split(key, 3)

    layer_sizes = [input_dim] + [width]*(depth-1) + [output_dim]
    generator = jpc.make_mlp(
        subkey[0], 
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=output_dim,
        act_fn=act_fn
    )
    # NOTE: input and output are inverted for the amortiser
    amortiser = jpc.make_mlp(
        subkey[1],
        input_dim=output_dim,
        width=width,
        depth=depth,
        output_dim=input_dim,
        act_fn=act_fn
    )
    
    gen_optim = optax.adam(lr)
    amort_optim = optax.adam(lr)
    optims = [gen_optim, amort_optim]
    
    gen_opt_state = gen_optim.init(
        (eqx.filter(generator, eqx.is_array), None)
    )
    amort_opt_state = amort_optim.init(eqx.filter(amortiser, eqx.is_array))
    opt_states = [gen_opt_state, amort_opt_state]

    train_loader, test_loader = get_mnist_loaders(batch_size)
    for iter, (img_batch, label_batch) in tqdm(enumerate(train_loader), total=len(train_loader)):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        result = jpc.make_hpc_step(
            generator=generator,
            amortiser=amortiser,
            optims=optims,
            opt_states=opt_states,
            input=label_batch,
            output=img_batch,
            max_t1=max_t1
        )
        generator, amortiser = result["generator"], result["amortiser"]
        gen_loss, amort_loss = result["losses"]
        if ((iter+1) % test_every) == 0 or (iter+1) == len(train_loader):
            amort_acc, hpc_acc, gen_acc, label_batch, img_preds = evaluate(
                key,
                layer_sizes,
                batch_size,
                generator,
                amortiser,
                test_loader
            )
            print(
                f"Iter {iter+1}, gen loss={gen_loss:4f}, "
                f"amort loss={amort_loss:4f}, "
                f"avg amort test accuracy={amort_acc:4f}, "
                f"avg hpc test accuracy={hpc_acc:4f}, "
                f"avg gen test accuracy={gen_acc:4f}, "
            )
            if (iter+1) >= n_train_iters:
                break

    fig = plot_mnist_imgs(img_preds, label_batch)
    if writter is not None:
        writter.add_figure(
            tag="HPC_generated_images",
            global_step=iter+1,
            figure=fig
        )
    plt.close(fig)
    return amortiser, generator
