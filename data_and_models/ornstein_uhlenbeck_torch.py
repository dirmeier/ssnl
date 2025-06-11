import ml_collections
import numpy as np
import torch
from jax import numpy as jnp

from data_and_models import ornstein_uhlenbeck


class model_fn:
    def __new__(cls, config):
        jax_data_fn, _, jax_simulator_fn, *_ = ornstein_uhlenbeck.model_fn(config)

        def data_fn(seed):
            y = jax_data_fn(seed)
            y = np.asarray(y).flatten()
            return torch.asarray(y, dtype=torch.float32, copy=True)

        def prior_fns():
            p = torch.distributions.Independent(
                torch.distributions.Uniform(0.0, torch.tensor([10.0, 5.0, 2.0])),
                1,
            )
            return p

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
    config.experiment_name = "ornstein_uhlenbeck"
    config.rng_key = 42
    config.model_fns = model_fn

    config.data = new_dict(
        ndim_data=100,
        t_end=10,
        ndim_theta=3,
    )

    return config
