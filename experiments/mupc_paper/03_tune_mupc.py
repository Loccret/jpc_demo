import os
import pickle
from typing import Any, Callable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
import optax
import optuna

from tqdm import tqdm

import jpc
from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import (
    setup_experiment,
    set_seed,
    init_weights
)

ROOT_PATH = 'tuning_results'
print("Tuning μPC for Classification")
os.makedirs(f"./{ROOT_PATH}", exist_ok=True)


def add_noise_to_activities(key: jax.random.PRNGKey, acts: List[jnp.ndarray], 
        sigma: float = 0.05, include_input: bool = True, 
        random_factor: float = 1.0) -> List[jnp.ndarray]:
    """Add Gaussian noise to forward pass activations and scale its radius"""
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


def evaluate(model, skip_model, test_loader, param_type):
    """Evaluate model on test set"""
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


def objective(trial):
    """Objective function for Optuna optimization"""
    
    # Fixed parameters
    seed = trial.number * 2
    dataset = "MNIST"
    d_in, d_out = 784, 10
    batch_size = 128
    max_epochs = 1
    param_type = "mupc"
    
    # Hyperparameters to tune
    width = 512
    n_hidden = 10
    act_fn = "relu"
    use_skips = True
    weight_init = "mupc"
    
    param_optim_id = "adam"
    param_lr = trial.suggest_float('param_lr', 1e-4, 1e-1, log=True)
    
    max_infer_iters = trial.suggest_int('max_infer_iters', 100, 500)
    activity_optim_id = 'gd'
    activity_lr = trial.suggest_float('activity_lr', 0.001, 10.0, log=True)
    
    sigma = 0.05
    random_factor = 0.5
    
    # Set seed
    set_seed(seed)
    key = jax.random.PRNGKey(seed)
    
    # Create model
    L = n_hidden + 1
    key, model_key, init_key = jr.split(key, 3)
    
    model = jpc.make_mlp(
        key=model_key,
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
            key=init_key,
            model=model,
            init_fn_id=weight_init,
            gain=gain
        )
    
    skip_model = jpc.make_skip_model(L) if use_skips else None
    
    # Create optimizers
    if param_optim_id == "sgd":
        param_optim = optax.sgd(param_lr)
    elif param_optim_id == "adam":
        param_optim = optax.adam(param_lr)
    else:
        raise ValueError("Invalid param optim id")
    
    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), skip_model)
    )
    
    activity_optim = optax.sgd(activity_lr) if (
        activity_optim_id == "gd"
    ) else optax.adam(activity_lr)
    
    # Get data
    train_loader, test_loader = get_dataloaders(dataset, batch_size)
    
    # Training loop
    iteration = 0
    TOTAL_STEPS = max_epochs * len(train_loader)
    FINAL_STEP = TOTAL_STEPS - 1
    
    pbar = tqdm(total=TOTAL_STEPS, leave=True, desc=f'Trial {trial.number}')
    
    for epoch in range(1, max_epochs + 1):
        for train_iter, (img_batch, label_batch) in enumerate(train_loader):
            img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
            
            # Initialize activities
            activities = jpc.init_activities_with_ffwd(
                model=model,
                input=img_batch,
                skip_model=skip_model,
                param_type=param_type
            )
            
            key, noise_key = jax.random.split(key)
            activities = add_noise_to_activities(
                key=noise_key,
                acts=activities,
                sigma=sigma,
                include_input=False,
                random_factor=random_factor
            )
            activity_opt_state = activity_optim.init(activities)
            
            # Inference
            for t in range(max_infer_iters):
                activity_update_result = jpc.update_activities(
                    params=(model, skip_model),
                    activities=activities,
                    optim=activity_optim,
                    opt_state=activity_opt_state,
                    output=label_batch,
                    input=img_batch,
                    param_type=param_type,
                    activity_decay=0,
                    weight_decay=0,
                    spectral_penalty=0
                )
                activities = activity_update_result["activities"]
                activity_opt_state = activity_update_result["opt_state"]
            
            # Update parameters
            param_update_result = jpc.update_params(
                params=(model, skip_model),
                activities=activities,
                optim=param_optim,
                opt_state=param_opt_state,
                output=label_batch,
                input=img_batch,
                param_type=param_type,
                activity_decay=0,
                weight_decay=0,
                spectral_penalty=0
            )
            model = param_update_result["model"]
            skip_model = param_update_result["skip_model"]
            param_opt_state = param_update_result["opt_state"]
            
            # Periodic evaluation
            if iteration % 100 == 0 or iteration == FINAL_STEP:
                valid_loss, valid_acc = evaluate(
                    model, skip_model, test_loader, param_type
                )
                pbar.set_postfix({
                    'Loss': f'{valid_loss:.4f}',
                    'Acc': f'{valid_acc:.4f}'
                })
                
                # Report intermediate value for pruning
                trial.report(valid_acc, iteration)
                
                # Handle pruning based on the intermediate value
                if trial.should_prune():
                    pbar.close()
                    raise optuna.exceptions.TrialPruned()
            
            pbar.update(1)
            iteration += 1
    
    pbar.close()
    
    # Final evaluation
    final_loss, final_acc = evaluate(
        model, skip_model, test_loader, param_type
    )
    
    return float(final_acc)


if __name__ == "__main__":
    device = jax.devices()[0]
    print(f"Device: {device}")
    
    # Create Optuna study with database storage
    study_name = "mupc_mnist_tuning"
    storage_name = f"sqlite:///./{ROOT_PATH}/{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(),
        storage=storage_name,
        direction="maximize",  # Maximize validation accuracy
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=100)
    )
    
    print(f"Study created: {study_name}")
    print(f"Storage: {storage_name}")
    
    # Run optimization
    N_TRIALS = 100  # Adjust based on computational budget
    
    def run_optimize(n_trials):
        for _ in range(n_trials):
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_name,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=100)
            )
            study.optimize(objective, n_trials=1, show_progress_bar=True, timeout=3600)  # 1 hour timeout per trial
    
    run_optimize(N_TRIALS)
    
    print(f"\nOptimization complete!")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (validation accuracy): {best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best params to file
    with open(f'./{ROOT_PATH}/best_params.pkl', 'wb') as f:
        pickle.dump(best_trial.params, f)
    print(f"\nBest parameters saved to ./{ROOT_PATH}/best_params.pkl")
