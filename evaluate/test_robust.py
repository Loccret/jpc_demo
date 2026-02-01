import jax
import jax.numpy as jnp
import equinox as eqx
from tqdm import tqdm
from robust_inference import perform_last_inference_by_acts, original_robust_inference_test
from structure import FCResNet
from core import mse_energy
from data_creator import get_mnist_loaders
import numpy as np

SEED = 4329

key = jax.random.PRNGKey(SEED)

INPUT_DIM = 784
WIDTH = 128
DEPTH = 100
OUTPUT_DIM = 10
ACT_FN = "relu"

ACTIVITY_LR = 5e-1
PARAM_LR = 1e-1
BATCH_SIZE = 64
TEST_EVERY = 100
N_TRAIN_ITERS = 900

mupc_model = FCResNet(
    key=key, 
    in_dim=INPUT_DIM, 
    width=WIDTH, 
    depth=DEPTH, 
    out_dim=OUTPUT_DIM, 
    act_fn=ACT_FN, 
    use_bias=False, 
    param_type="mupc"
)

save_path = "../models/mupc_resnet_depth100_mnist.eqx"
mupc = eqx.tree_deserialise_leaves(save_path, mupc_model)

train_mnist_loader, test_mnist_loader = get_mnist_loaders(batch_size=2000)

standard_robust_params = {
    'loader': test_mnist_loader,
    'model': mupc,
    'key': jax.random.PRNGKey(0),
    'lr': ACTIVITY_LR,
    'record_history': False,
    'ener_fn': mse_energy,
    'loss_fn': mse_energy,
    'feedforward_init': "ffwd",
    'sigma': 0.0,
    'RANDOM_FACTOR': 0.5,
    'act_amp': [1.0 for _ in range(len(mupc))],
    'temp_inference_step': 1000,
    'last_inference_step': 100,
    'INFERENCE_ROUNDS': 1,
    'writter': None,
    'noise_inference_scale': 2000.0
}

acc_by_act = {idx:[] for idx, _ in enumerate(mupc[:-1])}

for main_class in tqdm(range(10)):
    standard_robust_params['main_cls'] = main_class
    final_acts, new_labels = original_robust_inference_test(**standard_robust_params)
    for idx, _ in enumerate(final_acts[:-1]):
        acc = perform_last_inference_by_acts(mupc, final_acts, new_labels, key = jax.random.PRNGKey(standard_robust_params['main_cls']), act_idx=idx)
        acc_by_act[idx].append(acc)


for key, vals in acc_by_act.items():
    mean_acc = np.mean(np.array(vals))
    std_acc = np.std(np.array(vals))
    print(f'Act {key}: Mean Accuracy: {mean_acc:.4f}, Std Dev: {std_acc:.4f}')