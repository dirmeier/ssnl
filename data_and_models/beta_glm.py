"""Beta GLM model.

Adapted from the Bernoulli GLM from Lueckmann et al., AISTATS, 2021.
"""

import os
import pathlib

import jax
import ml_collections
import pandas as pd
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree
from tensorflow_probability.substrates.jax import distributions as tfd

from data_and_models import data_fn


class model_fn:
    def __new__(cls, config):
        M = config.ndim_theta - 1
        D = jnp.eye(M) - jnp.eye(M, k=-1)
        F = jnp.matmul(D, D) + jnp.diag(jnp.arange(M) / M) ** 0.5
        Binv = jnp.zeros((M + 1, M + 1))
        Binv = Binv.at[0, 0].set(0.5)
        Binv = Binv.at[1:, 1:].set(jnp.matmul(F.T, F))
        B = jnp.linalg.inv(Binv)
        # design matrix
        dir = pathlib.Path(__file__).parent.resolve()
        fl = os.path.join(dir, "beta_glm_design_matrix.pkl")
        design_matrix = pd.read_pickle(fl)
        concentration = 50.0

        prior = tfd.MultivariateNormalFullCovariance(loc=0.0, covariance_matrix=B)
        init_params = prior.sample(seed=jr.PRNGKey(0))
        _, unravel_fn = ravel_pytree(init_params)

        def _simulate(seed, theta):
            eta = (design_matrix @ theta.T).T
            mean = jax.nn.sigmoid(eta)
            mean = mean.clip(0.0, 1.0)
            y = tfd.Beta(mean * concentration, (1.0 - mean) * concentration).sample(seed=seed)
            return y

        def prior_fns():
            return prior.sample, prior.log_prob

        def simulator_fn(config):
            return _simulate

        def unnormalized_posterior():
            def _likelihood_fn(theta, y):
                eta = (design_matrix @ theta.T).T
                mean = jax.nn.sigmoid(eta)
                distr = tfd.Independent(
                    tfd.Beta(mean * concentration, (1.0 - mean) * concentration), 1
                )
                lp = distr.log_prob(y)
                return lp

            def log_density_fn(theta, y):
                prior_lp = prior.log_prob(theta)
                likelihood_lp = _likelihood_fn(theta, y)

                lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
                return lp

            return log_density_fn

        return (
            data_fn(config, _simulate, prior),
            prior_fns,
            simulator_fn(config),
            unnormalized_posterior,
        )


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
