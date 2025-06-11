"""Jansen-Rit neural mass model.

Adopted from Rodriguez, et al., NeurIPS, 2021.
"""

import ml_collections
from jax import numpy as jnp
from jax import random as jr
from jrnmm import simulate as simulate_jrnmm
from tensorflow_probability.substrates.jax import distributions as tfd

from data_and_models import data_fn


class model_fn:
    def __new__(cls, config):
        len_simul = 1024
        len_desired = 100
        len_subset = len_simul - len_simul % len_desired
        subset_idxs = jnp.arange(0, len_subset, len_subset // len_desired)

        prior = tfd.Independent(
            tfd.Uniform(
                jnp.array([10.0, 50.0, 100.0, -20.0]), jnp.array([250.0, 500.0, 5000.0, 20.0])
            ),
            1,
        )

        def _simulate(seed, theta):
            Cs, mus, sigmas, gains = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
            y = simulate_jrnmm(
                seed,
                dt=1 / 512,
                t_end=config.t_end,
                initial_states=jnp.array([0.08, 18, 15, -0.5, 0, 0]),
                Cs=Cs,
                mus=mus,
                sigmas=sigmas,
                gains=gains,
            )
            return y[:, subset_idxs, 0]

        def prior_fns():
            return prior.sample, prior.log_prob

        def simulator_fn(config):
            return _simulate

        return data_fn(config, _simulate, prior), prior_fns, simulator_fn(config), None


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "jansen_rit_strang"
    config.rng_key = 42
    config.model_fns = model_fn

    config.data = new_dict(
        ndim_data=100,
        t_end=8,
        ndim_theta=4,
    )

    return config
