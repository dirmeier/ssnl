import ml_collections
import numpy as np
import torch
from jax import numpy as jnp

from data_and_models import beta_glm


class model_fn:
    def __new__(cls, config):
        # from SBIBM: https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/bernoulli_glm/task.py
        M = config.ndim_theta - 1
        D = torch.diag(torch.ones(M)) - torch.diag(torch.ones(M - 1), -1)
        F = torch.matmul(D, D) + torch.diag(1.0 * torch.arange(M) / (M)) ** 0.5
        Binv = torch.zeros(size=(M + 1, M + 1))
        Binv[0, 0] = 0.5  # offset
        Binv[1:, 1:] = torch.matmul(F.T, F)  # filter
        prior_params = {"loc": torch.zeros((M + 1,)), "precision_matrix": Binv}
        prior = torch.distributions.MultivariateNormal(**prior_params)

        jax_data_fn, _, jax_simulator_fn, *_ = beta_glm.model_fn(config)

        def data_fn(seed):
            y = jax_data_fn(seed)
            y = np.asarray(y).flatten()
            return torch.asarray(y, dtype=torch.float32, copy=True)

        def prior_fns():
            return prior

        def simulator(seed, theta):
            theta = jnp.asarray(theta.numpy())
            y = jax_simulator_fn(seed, theta)
            y = np.asarray(y)
            y = torch.asarray(y, dtype=torch.float32, copy=True)
            return y

        return data_fn, prior_fns, simulator, None


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "beta_glm"
    config.rng_key = 42
    config.model_fns = model_fn

    config.data = new_dict(
        ndim_data=100,
        ndim_theta=10,
    )

    return config
