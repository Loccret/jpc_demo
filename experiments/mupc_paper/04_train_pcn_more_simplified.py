import os
from jaxtyping import PRNGKeyArray, PyTree, ArrayLike, Scalar, Array
from typing import Any, Callable, List, Optional, Tuple, Dict, Literal
import jax
import jax.random as jr
import jax.numpy as jnp
from tqdm import tqdm
import equinox as eqx
import jpc
import optax
from jpc._core._energies import pc_energy_fn
from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import (
    setup_experiment,
    set_seed,
    init_weights
)


def add_noise_to_activities(key: PRNGKeyArray, acts: List[Array], 
        sigma: Scalar = 0.05,include_input: bool = True, 
        random_factor: float = 1.0)-> List[Array]:
    """
        add Gaussian noise to forward pass activations and scale it's radius
    """
    random_keys = jax.random.split(key, len(acts))
    random_activities = []
    for act, rand_key in zip(acts, random_keys):
        ran_real = random_factor * sigma * jax.random.normal(rand_key, shape=act.shape) + (1 - random_factor) * act
        ran_act = ran_real
        radii = jnp.linalg.norm(act, axis=-1, keepdims=True)
        ran_act = ran_act / (jnp.linalg.norm(ran_act, axis=-1, keepdims=True) + 1e-12) * radii
        random_activities.append(ran_act)
    if include_input:
        random_activities.insert(0, acts[0])
    return random_activities



def evaluate(params, test_loader, param_type):
    model, skip_model = params
    avg_test_loss, avg_test_acc = 0, 0
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        test_loss, test_acc = jpc.test_discriminative_pc(
            model=model,
            output=label_batch,
            input=img_batch,
            skip_model=skip_model,
            param_type=param_type
        )
        avg_test_loss += test_loss
        avg_test_acc += test_acc

    return avg_test_loss / len(test_loader), avg_test_acc / len(test_loader)


def train_mlp(
        seed,
        dataset,
        width,
        n_hidden,
        act_fn,
        use_skips,
        weight_init,
        param_type,
        param_optim_id,
        param_lr,
        batch_size,
        max_infer_iters,
        activity_optim_id,
        activity_lr,
        activity_decay,
        weight_decay,
        spectral_penalty,
        max_epochs,
        test_every,
        save_dir
):
    set_seed(seed)
    key = jax.random.PRNGKey(seed)
    keys = jr.split(key, 4)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # create and initialise model
    d_in, d_out = 784, 10
    L = n_hidden + 1
    model = jpc.make_mlp(
        key=keys[0],
        input_dim=d_in,
        width=width,
        depth=L,
        output_dim=d_out,
        act_fn=act_fn,
        use_bias=False,
        param_type=param_type
    )
    if weight_init == "orthogonal":
        gain = 1.05 if (weight_init == "orthogonal" and act_fn == "tanh") else 1
        model = init_weights(
            key=keys[1],
            model=model,
            init_fn_id=weight_init,
            gain=gain
        )
    skip_model = jpc.make_skip_model(L) if use_skips else None

    # optimisers
    if param_optim_id == "sgd":
        param_optim = optax.sgd(param_lr)
    elif param_optim_id == "adam":
        param_optim = optax.adam(param_lr)
    else:
        raise ValueError("Invalid param optim id. Options are 'sgd' and 'adam'.")

    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), skip_model)
    )
    activity_optim = optax.sgd(activity_lr) if (
            activity_optim_id == "gd"
    ) else optax.adam(activity_lr)

    # data
    iteration = 0
    train_loader, test_loader = get_dataloaders(dataset, batch_size)

    energy_history = []
    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}\n-------------------------------")

        for train_iter, (img_batch, label_batch) in enumerate(tqdm(train_loader, total = len(train_loader))):
            img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
            # if iteration == 5:
            #     jax.profiler.start_trace("./runs/jax_profile")
            # if iteration == 10:
            #     jax.profiler.stop_trace()
            # initialise activities
            activities = jpc.init_activities_with_ffwd(
                model=model,
                input=img_batch,
                skip_model=skip_model,
                param_type=param_type
            )

            # key, noise_key = jax.random.split(key)
            # activities = add_noise_to_activities(
            #     key=noise_key,
            #     acts=activities,
            #     sigma=0.05,
            #     include_input=False,
            #     random_factor=0.5
            # )
            activity_opt_state = activity_optim.init(activities)

            # inference
            for t in range(max_infer_iters):
                activity_update_result = jpc.update_activities(
                    params=(model, skip_model),
                    activities=activities,
                    optim=activity_optim,
                    opt_state=activity_opt_state,
                    output=label_batch,
                    input=img_batch,
                    param_type=param_type,
                    activity_decay=activity_decay,
                    weight_decay=weight_decay,
                    spectral_penalty=spectral_penalty
                )
                activities = activity_update_result["activities"]
                activity_opt_state = activity_update_result["opt_state"]

                step_energy = pc_energy_fn(
                    params = (model, skip_model),
                    activities = activities,
                    y = label_batch,
                    x = img_batch,
                    param_type = param_type,
                    weight_decay = weight_decay,
                    spectral_penalty = spectral_penalty,
                    activity_decay = activity_decay,
                    record_history = True
                )
                energy_history.append(step_energy)

            # update parameters
            param_update_result = jpc.update_params(
                params=(model, skip_model),
                activities=activities,
                optim=param_optim,
                opt_state=param_opt_state,
                output=label_batch,
                input=img_batch,
                param_type=param_type,
                activity_decay=activity_decay,
                weight_decay=weight_decay,
                spectral_penalty=spectral_penalty
            )
            model = param_update_result["model"]
            skip_model = param_update_result["skip_model"]
            param_opt_state = param_update_result["opt_state"]
            iteration += 1

        # Evaluate after each epoch
        avg_test_loss, avg_test_acc = evaluate(
            params=(model, skip_model),
            test_loader=test_loader,
            param_type=param_type
        )
        print(f"Epoch {epoch} - Test Accuracy: {avg_test_acc:.4f}\n")


    # save energy history
    try:
        energy_history = jnp.array(energy_history)
        jnp.save(os.path.join(save_dir, "energy_history.npy"), energy_history)
    except Exception as e:
        print(f"Error saving energy history: {e}")
    # save model
    if save_dir is not None:
        eqx.tree_serialise_leaves(os.path.join(save_dir, "final_model.eqx"), model)
        if skip_model is not None:
            eqx.tree_serialise_leaves(os.path.join(save_dir, "final_skip_model.eqx"), skip_model)


if __name__ == "__main__":
    device = jax.devices()[0]
    print(f"device: {device}")
    
    # # higher precision for more accurate inversion of the activity Hessian
    # jax.config.update("jax_enable_x64", True)

    # Hardcoded hyperparameters
    results_dir = "pcn_results"
    dataset = "MNIST"
    loss_id = "mse"
    act_fn = "relu"
    width = 512
    n_hidden = 128
    use_skips = True
    weight_init = "standard_gauss"
    param_type = "mupc"
    # param_type = "sp"
    param_optim_id = "adam"
    param_lr = 1e-1
    batch_size = 128
    # max_infer_iters = 128
    max_infer_iters = 0
    activity_optim_id = "gd"
    # activity_optim_id = "adam"
    activity_lr = 5e-1
    activity_decay = 0
    weight_decay = 0
    spectral_penalty = 0
    max_epochs = 1
    test_every = 300
    n_seeds = 1

    for seed in range(n_seeds):
        # save_dir = setup_experiment(
        #     results_dir=results_dir,
        #     dataset=dataset,
        #     loss_id=loss_id,
        #     width=width,
        #     n_hidden=n_hidden,
        #     act_fn=act_fn,
        #     use_skips=use_skips,
        #     weight_init=weight_init,
        #     param_type=param_type,
        #     param_optim_id=param_optim_id,
        #     param_lr=param_lr,
        #     batch_size=batch_size,
        #     max_infer_iters=max_infer_iters,
        #     activity_optim_id=activity_optim_id,
        #     activity_lr=activity_lr,
        #     activity_decay=activity_decay,
        #     weight_decay=weight_decay,
        #     spectral_penalty=spectral_penalty,
        #     max_epochs=max_epochs,
        #     seed=seed
        # )
        # print(f"Training with seed {seed} and saving results to {save_dir}")
        train_mlp(
            seed=seed,
            dataset=dataset,
            width=width,
            n_hidden=n_hidden,
            act_fn=act_fn,
            use_skips=use_skips,
            weight_init=weight_init,
            param_type=param_type,
            param_optim_id=param_optim_id,
            param_lr=param_lr,
            batch_size=batch_size,
            max_infer_iters=max_infer_iters,
            activity_optim_id=activity_optim_id,
            activity_lr=activity_lr,
            activity_decay=activity_decay,
            weight_decay=weight_decay,
            spectral_penalty=spectral_penalty,
            max_epochs=max_epochs,
            test_every=test_every,
            # save_dir=save_dir
            save_dir = None
        )
