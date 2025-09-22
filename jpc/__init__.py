import importlib.metadata

from ._core import (
    init_activities_with_ffwd as init_activities_with_ffwd,
    init_activities_from_normal as init_activities_from_normal,
    init_activities_with_amort as init_activities_with_amort,
    pc_energy_fn as pc_energy_fn,
    hpc_energy_fn as hpc_energy_fn,
    bpc_energy_fn as bpc_energy_fn,
    _get_param_scalings as _get_param_scalings,
    neg_activity_grad as neg_activity_grad,
    solve_inference as solve_inference,
    compute_activity_grad as compute_activity_grad,
    compute_pc_param_grads as compute_pc_param_grads,
    compute_hpc_param_grads as compute_hpc_param_grads,
    compute_bpc_activity_grad as compute_bpc_activity_grad,
    compute_bpc_param_grads as compute_bpc_param_grads,
    update_activities as update_activities,
    update_params as update_params,
    update_bpc_activities as update_bpc_activities,
    update_bpc_params as update_bpc_params,
    compute_linear_equilib_energy as compute_linear_equilib_energy,
    compute_linear_activity_hessian as compute_linear_activity_hessian,
    compute_linear_activity_solution as compute_linear_activity_solution,
    _check_param_type as _check_param_type
)
from ._utils import (
    make_mlp as make_mlp,
    make_basis_mlp as make_basis_mlp,
    make_skip_model as make_skip_model,
    get_act_fn as get_act_fn,
    mse_loss as mse_loss,
    cross_entropy_loss as cross_entropy_loss,
    compute_accuracy as compute_accuracy,
    get_t_max as get_t_max,
    compute_activity_norms as compute_activity_norms,
    compute_infer_energies as compute_infer_energies,
    compute_param_norms as compute_param_norms
)
from ._train import (
    make_pc_step as make_pc_step,
    make_hpc_step as make_hpc_step
)
from ._test import (
    test_discriminative_pc as test_discriminative_pc,
    test_generative_pc as test_generative_pc,
    test_hpc as test_hpc
)


__version__ = importlib.metadata.version("jpc")
