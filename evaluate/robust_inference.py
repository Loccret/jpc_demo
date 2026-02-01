from typing import Any, Callable, List, Optional, Tuple, Dict, Literal
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, Scalar
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from core import (compute_ce_loss, init_activities_from_normal_ffwd, forward_and_extract_activities, compute_energy, perform_inference, validate, make_modern_mlp, mse_energy)
from data_creator import get_mnist_loaders

def _prepare_data_for_robust_test(loader, main_cls: int, num_samples: int = 100, key = None):
    """Extract and organize data by class. This runs outside of JIT."""
    # Extract all data from loader
    all_imgs = []
    all_labels = []
    for images, labels in loader:
        all_imgs.append(jnp.array(images.numpy()))
        all_labels.append(jnp.array(labels.numpy()))
    
    new_imgs = jnp.concatenate(all_imgs, axis=0)
    new_labels = jnp.concatenate(all_labels, axis=0)
    
    # Get class indices
    label_classes = jnp.argmax(new_labels, axis=-1)
    
    # Pre-extract data for each class
    class_data = {}
    for cls in range(10):
        all_indices = jnp.where(label_classes == cls)[0]
        
        if key is not None:
            # Randomly select indices
            key, subkey = jax.random.split(key)
            if len(all_indices) >= num_samples:
                indices = jax.random.choice(subkey, all_indices, shape=(num_samples,), replace=False)
            else:
                # If not enough samples, use all and randomly sample with replacement to fill
                indices = jax.random.choice(subkey, all_indices, shape=(num_samples,), replace=True)
        else:
            # Take first num_samples
            indices = all_indices[:num_samples]
            # Pad if needed to ensure fixed size
            if len(indices) < num_samples:
                indices = jnp.pad(indices, (0, num_samples - len(indices)), mode='edge')
        
        class_data[cls] = {
            'imgs': new_imgs[indices],
            'labels': new_labels[indices]
        }
    
    return class_data


@eqx.filter_jit
def _jitted_robust_inference_core(
    model: List[Callable], 
    main_cls_imgs: Array,
    other_cls_imgs: Array,  # Shape: (9, num_samples, ...)
    other_cls_labels: Array,
    fin_imgs: Array,
    fin_labels: Array,
    key: PRNGKeyArray,
    lr: float,
    act_amp: List[float],
    temp_inference_step: int,
    last_inference_step: int,
    INFERENCE_ROUNDS: int,
    record_history: bool,
    ener_fn: Callable,
    loss_fn: Callable,
    feedforward_init: str,
    sigma: float, 
    noise_inference_scale: float = 5.0
):
    """JIT-compiled core inference logic without boolean indexing."""
    inference = True
    
    # Initialize activations
    if feedforward_init == "ffwd":
        activations = forward_and_extract_activities(model, main_cls_imgs, key=key, inference=inference)
    else:  # "random"
        activations = init_activities_from_normal_ffwd(
            key, model, input_shape=main_cls_imgs.shape, sigma=sigma, 
            inference=inference, include_input=False)
    
    activations = [amp * act for amp, act in zip(act_amp, activations)]
    
    
    # Stack all other class data for all rounds
    num_other_classes = other_cls_imgs.shape[0]
    
    for _ in range(INFERENCE_ROUNDS):
        for cls_idx in range(num_other_classes):
            activations, _ = perform_inference(
                model, activations, key, yy=None, xx=other_cls_imgs[cls_idx], 
                num_iters=temp_inference_step, lr=noise_inference_scale * lr, record_history=False, 
                ener_fn=ener_fn, loss_fn=loss_fn, inference=inference
            )
    
    # Final inference on main class
    final_act, energy_history = perform_inference(
        model, activations, key, yy=None, xx=fin_imgs, 
        num_iters=last_inference_step, lr=1 * lr, record_history=record_history, 
        ener_fn=ener_fn, loss_fn=loss_fn, inference=inference
    )
    
    return final_act, fin_labels




def original_robust_inference_test(
    loader: Any, 
    model: List[Callable], 
    key: PRNGKeyArray, 
    lr: float = 0.05, 
    record_history: bool = False, 
    ener_fn: Callable = mse_energy, 
    loss_fn: Callable = mse_energy, 
    feedforward_init: str = "ffwd", 
    sigma: float = 0.05, 
    RANDOM_FACTOR: float = 1.0, 
    act_amp: List[float] = [0.1, 0.1, 0.1], 
    temp_inference_step: int = 5, 
    last_inference_step: int = 100, 
    INFERENCE_ROUNDS: int = 3, 
    main_cls: int = 0, 
    writter: Any = None,
    noise_inference_scale: float = 5.0
):
    """Wrapper function that prepares data and calls JIT-compiled core."""
    print(f"noise_inference_scale is: {noise_inference_scale}")
    
    # Prepare data outside of JIT
    key, data_select_key = jax.random.split(key)
    class_data = _prepare_data_for_robust_test(
        loader, main_cls, num_samples=100, key=data_select_key)
    
    # Separate main class from others
    main_cls_imgs = class_data[main_cls]['imgs']
    
    # Stack other class data
    other_classes = [cls for cls in range(10) if cls != main_cls]
    other_cls_imgs = jnp.stack([class_data[cls]['imgs'] for cls in other_classes])
    other_cls_labels = jnp.stack([class_data[cls]['labels'] for cls in other_classes])
    
    # Call JIT-compiled function
    final_act, new_labels = _jitted_robust_inference_core(
        model=model,
        main_cls_imgs=main_cls_imgs,
        other_cls_imgs=other_cls_imgs,
        other_cls_labels=other_cls_labels,
        fin_imgs=main_cls_imgs,
        fin_labels=class_data[main_cls]['labels'],
        key=key,
        lr=lr,
        act_amp=act_amp,
        temp_inference_step=temp_inference_step,
        last_inference_step=last_inference_step,
        INFERENCE_ROUNDS=INFERENCE_ROUNDS,
        record_history=record_history,
        ener_fn=ener_fn,
        loss_fn=loss_fn,
        feedforward_init=feedforward_init,
        sigma=sigma,
        noise_inference_scale=noise_inference_scale
    )
    
    # Logging (done outside JIT)
    if writter is not None:
        energies = compute_energy(model, final_act, key, y=None, x=main_cls_imgs, 
                                ener_fn=ener_fn, inference=True)
        for i, (act, energy) in enumerate(zip(final_act, energies)):
            writter.add_scalar(f'Energy/Layer_{i}', float(energy), 0)
            act_norm = jnp.linalg.norm(act, axis=-1).mean().item()
            writter.add_scalar(f'Act_Norm/Layer_{i}', act_norm, 0)
    
    return final_act, new_labels


def perform_last_inference_by_acts(model: List[Callable], activations: List[Array], labels: Array, key: PRNGKeyArray, act_idx: int):
    act = activations[act_idx]
    for i in range(act_idx+1, len(model)):
        vlayer = eqx.filter_vmap(
            lambda x_single: model[i](x_single, key=key, inference=True), 
            in_axes=0, out_axes=0
        )
        act = vlayer(act)
    pred_classes = jnp.argmax(act, axis=-1)
    true_classes = jnp.argmax(labels, axis=-1)
    acc = jnp.mean(pred_classes == true_classes)
    return acc