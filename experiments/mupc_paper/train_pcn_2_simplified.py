import os
import pickle
import numpy as np

import jax
import jax.random as jr
import jax.numpy as jnp
from jax.tree_util import tree_map

import equinox as eqx
import jpc
import optax
from optimistix import rms_norm

from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import (
    setup_experiment,
    set_seed,
    init_weights,
    compute_param_l2_norms,
    compute_param_spectral_norms,
    compute_hessian_eigens,
    compute_cond_num
)
from experiments.mupc_paper.plotting import (
    plot_loss,
    plot_loss_and_accuracy,
    plot_n_infer_iters,
    plot_norms,
    plot_energies,
    plot_hessian_eigenvalues_during_training,
    plot_max_min_eigenvals,
    plot_max_min_eigenvals_2_axes,
    plot_metric_stats
)


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
    train_loader, test_loader = get_dataloaders(dataset, batch_size)

    # metrics
    train_losses = []
    test_losses, test_accs = [], []

    n_train_iters = len(train_loader.dataset) // batch_size * max_epochs
    n_test_iters = n_train_iters // test_every * max_epochs
    layer_idxs = [0, int(L / 4) - 1, int(L / 2) - 1, int(L * 3 / 4) - 1, L - 1]

    mean_abs_activities = np.zeros(
        (n_train_iters, max_infer_iters + 1, len(layer_idxs))
    )
    activity_l2_norms = np.zeros_like(mean_abs_activities)
    n_infer_iters = np.ones(n_train_iters) * max_infer_iters

    param_l2_norms = np.zeros((n_train_iters + 1, len(layer_idxs)))
    param_spectral_norms = np.zeros_like(param_l2_norms)

    train_num_energies = np.zeros((n_test_iters + 1, len(layer_idxs)))

    has_diverged, no_learning = False, False
    global_batch_id = 0
    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}\n-------------------------------")

        for train_iter, (img_batch, label_batch) in enumerate(train_loader):
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

            # record metrics at init
            i = 0
            for l, act in enumerate(activities):
                if l in layer_idxs:
                    mean_abs_activities[global_batch_id, 0, i] = np.array(
                        jnp.mean(jnp.abs(act))
                    )
                    activity_l2_norms[global_batch_id, 0, i] = np.array(
                        jnp.linalg.norm(act, axis=1, ord=2).mean()
                    )
                    i += 1

            if global_batch_id == 0:
                param_l2_norms[0] = compute_param_l2_norms(
                    model=model,
                    act_fn=act_fn,
                    layer_idxs=layer_idxs
                )
                param_spectral_norms[0] = compute_param_spectral_norms(
                    model=model,
                    act_fn=act_fn,
                    layer_idxs=layer_idxs
                )

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
                activity_grads = activity_update_result["grads"]
                if rms_norm(activity_grads) < 1e-3 + 1e-3 * rms_norm(activity_grads):
                    n_infer_iters[global_batch_id] = t            
                
                if global_batch_id == 0 or global_batch_id % test_every == 0:
                    num_energies = jpc.pc_energy_fn(
                        params=(model, skip_model),
                        activities=activities,
                        y=label_batch,
                        x=img_batch,
                        param_type=param_type,
                        activity_decay=activity_decay,
                        weight_decay=weight_decay,
                        spectral_penalty=spectral_penalty,
                        record_layers=True
                    )
                    test_iter = 0 if (
                            global_batch_id == 0
                    ) else int(global_batch_id / test_every)
                    train_num_energies[test_iter] = np.array([
                        e for l, e in enumerate(reversed(num_energies)) if l in layer_idxs
                    ])

                i = 0
                for l, act in enumerate(activities):
                    if l in layer_idxs:
                        mean_abs_activities[global_batch_id, t + 1, i] = np.array(
                            jnp.mean(jnp.abs(act))
                        )
                        activity_l2_norms[global_batch_id, t + 1, i] = np.array(
                            jnp.linalg.norm(act, axis=1, ord=2).mean()
                        )
                        i += 1

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

            param_l2_norms[global_batch_id + 1] = compute_param_l2_norms(
                model=model,
                act_fn=act_fn,
                layer_idxs=layer_idxs
            )
            param_spectral_norms[global_batch_id + 1] = compute_param_spectral_norms(
                model=model,
                act_fn=act_fn,
                layer_idxs=layer_idxs
            )
            train_losses.append(train_loss)
            global_batch_id += 1

            if global_batch_id % test_every == 0:
                print(
                    f"Train loss: {train_loss:.7f} [{train_iter * len(img_batch)}/{len(train_loader.dataset)}]"
                )
                avg_test_loss, avg_test_acc = evaluate(
                    params=(model, skip_model),
                    test_loader=test_loader,
                    param_type=param_type
                )
                test_losses.append(avg_test_loss)
                test_accs.append(avg_test_acc)
                print(f"Avg test accuracy: {avg_test_acc:.4f}\n")

            if np.isinf(train_loss) or np.isnan(train_loss):
                has_diverged = True
                break
            
            if global_batch_id >= test_every and avg_test_acc < 15:
                no_learning = True
                break
        
        if has_diverged:
            print(
                f"Stopping training because of diverging loss: {train_loss}"
            )
            break

        if no_learning:
            print(
                f"Stopping training because of chance accuracy (no learning): {avg_test_acc}"
            )
            break

    plot_loss(
        loss=train_losses,
        yaxis_title="Train loss",
        xaxis_title="Iteration",
        save_path=f"{save_dir}/train_losses.pdf"
    )
    plot_loss_and_accuracy(
        loss=test_losses,
        accuracy=test_accs,
        mode="test",
        xaxis_title="Training iteration",
        save_path=f"{save_dir}/test_losses_and_accs.pdf",
        test_every=test_every
    )
    plot_n_infer_iters(
        n_infer_iters=n_infer_iters,
        save_path=f"{save_dir}/n_infer_iters.pdf"
    )
    plot_norms(
        norms=param_l2_norms,
        norm_type="param_l2",
        save_path=f"{save_dir}/param_l2_norms.pdf"
    )
    plot_norms(
        norms=param_spectral_norms,
        norm_type="param_spectral",
        save_path=f"{save_dir}/param_spectral_norms.pdf"
    )
    plot_energies(
        energies=train_num_energies.T,
        test_every=test_every,
        save_path=f"{save_dir}/energies.pdf",
        log=False
    )
    plot_energies(
        energies=train_num_energies.T,
        test_every=test_every,
        save_path=f"{save_dir}/log_energies.pdf",
        log=True
    )
    plot_norms(
        norms=activity_l2_norms[0],
        norm_type="activity",
        save_path=f"{save_dir}/activity_l2_norms_at_init.pdf"
    )
    plot_norms(
        norms=activity_l2_norms[0],
        norm_type="activity",
        save_path=f"{save_dir}/log_activity_l2_norms_at_init.pdf",
        log=True
    )
    plot_norms(
        norms=activity_l2_norms[-1],
        norm_type="activity",
        save_path=f"{save_dir}/activity_l2_norms_last_train_iter.pdf"
    )
    plot_norms(
        norms=activity_l2_norms[-1],
        norm_type="activity",
        save_path=f"{save_dir}/log_activity_l2_norms_last_train_iter.pdf",
        log=True
    )

    np.save(f"{save_dir}/batch_train_losses.npy", train_losses)
    np.save(f"{save_dir}/test_losses.npy", test_losses)
    np.save(f"{save_dir}/test_accs.npy", test_accs)
    np.save(f"{save_dir}/num_energies.npy", train_num_energies)

    np.save(f"{save_dir}/mean_abs_activities.npy", mean_abs_activities)
    np.save(f"{save_dir}/activity_l2_norms.npy", activity_l2_norms)
    np.save(f"{save_dir}/n_infer_iters.npy", n_infer_iters)

    np.save(f"{save_dir}/param_l2_norms.npy", param_l2_norms)
    np.save(f"{save_dir}/param_spectral_norms.npy", param_spectral_norms)

    return test_accs


if __name__ == "__main__":
    device = jax.devices()[0]
    print(f"device: {device}")
    
    # higher precision for more accurate inversion of the activity Hessian
    jax.config.update("jax_enable_x64", True)

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
    param_optim_id = "adam"
    param_lr = 1e-1
    batch_size = 64
    max_infer_iters = 128
    activity_optim_id = "gd"
    activity_lr = 5e-1
    activity_decay = 0
    weight_decay = 0
    spectral_penalty = 0
    max_epochs = 5
    test_every = 300
    n_seeds = 5

    test_accs_seeds = [[] for _ in range(n_seeds)]
    for seed in range(n_seeds):
        save_dir = setup_experiment(
            results_dir=results_dir,
            dataset=dataset,
            loss_id=loss_id,
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
            seed=seed
        )
        test_accs = train_mlp(
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
            save_dir=save_dir
        )
        test_accs_seeds[seed] = test_accs

    plot_metric_stats(
        metric=test_accs_seeds,
        metric_id="test_acc",
        test_every=test_every,
        save_path=f"{save_dir[:-1]}/test_accs.pdf"
    )
