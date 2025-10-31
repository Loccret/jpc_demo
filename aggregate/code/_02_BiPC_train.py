import jpc

import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import optax
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from _01_utilities import get_mnist_loaders, plot_mnist_imgs
import io
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import warnings




def evaluate(generator, amortiser, test_loader):
    amort_accs = 0.
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        preds = jpc.init_activities_with_ffwd(
            model=amortiser[::-1],
            input=img_batch
        )[-1]
        amort_accs += jpc.compute_accuracy(label_batch, preds)

    img_preds = jpc.init_activities_with_ffwd(
        model=generator,
        input=label_batch
    )[-1]

    return (
        amort_accs / len(test_loader),
        label_batch,
        img_preds
    )



def train_BiPC(
      seed,
      input_dim,
      width,
      depth,
      output_dim,
      act_fn,
      batch_size,
      activity_lr,
      param_lr,
      test_every,
      n_train_iters,
      writter = None
):
    key = jax.random.PRNGKey(seed)
    gen_key, amort_key = jax.random.split(key, 2)

    # models (NOTE: input and output are inverted for the amortiser)
    layer_sizes = [input_dim] + [width]*(depth-1) + [output_dim]
    generator = jpc.make_mlp(
        gen_key, 
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=output_dim,
        act_fn=act_fn
    )
    amortiser = jpc.make_mlp(
        amort_key,
        input_dim=output_dim,
        width=width,
        depth=depth,
        output_dim=input_dim,
        act_fn=act_fn
    )[::-1]
        
    # optimisers
    activity_optim = optax.sgd(activity_lr)
    gen_optim = optax.adam(param_lr)
    amort_optim = optax.adam(param_lr)
    optims = [gen_optim, amort_optim]
    
    gen_opt_state = gen_optim.init(eqx.filter(generator, eqx.is_array))
    amort_opt_state = amort_optim.init(eqx.filter(amortiser, eqx.is_array))
    opt_states = [gen_opt_state, amort_opt_state]

    # data
    train_loader, test_loader = get_mnist_loaders(batch_size)

    for iter, (img_batch, label_batch) in enumerate(train_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
        
        # discriminative loss
        amort_activities = jpc.init_activities_with_ffwd(
            model=amortiser[::-1],
            input=img_batch
        )
        amort_loss = jpc.mse_loss(amort_activities[-1], label_batch)

        # generative loss & initialisation
        activities = jpc.init_activities_with_ffwd(
            model=generator,
            input=label_batch
        )
        gen_loss = jpc.mse_loss(activities[-1], img_batch)
        activity_opt_state = activity_optim.init(activities)

        # inference
        for t in range(depth-1):
            activity_update_result = jpc.update_bpc_activities(
                top_down_model=generator,
                bottom_up_model=amortiser,
                activities=activities,
                optim=activity_optim,
                opt_state=activity_opt_state,
                output=img_batch,
                input=label_batch
            )
            activities = activity_update_result["activities"]
            activity_opt_state = activity_update_result["opt_state"]

        # learning
        param_update_result = jpc.update_bpc_params(
            top_down_model=generator,
            bottom_up_model=amortiser,
            activities=activities,
            top_down_optim=gen_optim,
            bottom_up_optim=amort_optim,
            top_down_opt_state=gen_opt_state,
            bottom_up_opt_state=amort_opt_state,
            output=img_batch,
            input=label_batch
        )
        generator, amortiser = param_update_result["models"]
        gen_opt_state, amort_opt_state  = param_update_result["opt_states"]

        if ((iter+1) % test_every) == 0:
            amort_acc, label_batch, img_preds = evaluate(
                generator,
                amortiser,
                test_loader
            )
            print(
                f"Iter {iter+1}, gen loss={gen_loss:4f}, "
                f"amort loss={amort_loss:4f}, "
                f"avg amort test accuracy={amort_acc:4f}"
            )

            if (iter+1) >= n_train_iters:
                break
                
    fig = plot_mnist_imgs(img_preds, label_batch)
    # log to tensorboard
    if writter is not None:
        writter.add_figure(
            tag="BiPC_generated_images",
            figure=fig,
            global_step=iter+1
        )
    plt.close(fig)


