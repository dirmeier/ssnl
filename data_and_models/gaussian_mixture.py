"""Gaussian mixture model.

Adopted from Lueckmann et al., AISTATS, 2021.
"""

import distrax
import ml_collections
from jax import numpy as jnp
from jax import random as jr

from data_and_models import data_fn


class model_fn:
    def __new__(cls, config):
        prior = distrax.Independent(
            distrax.Uniform(jnp.full(config.ndim_theta, -10.0), jnp.full(config.ndim_theta, 10.0)),
            1,
        )
        scales_params = jnp.array([1.0, 0.1])

        def _simulate(seed, theta):
            mix_key, data_key = jr.split(seed)
            idxs = jr.bernoulli(mix_key, shape=(theta.shape[0],)).astype(jnp.int32)
            scales = scales_params[idxs].reshape(-1, 1)
            y = distrax.Normal(theta, scales).sample(seed=data_key)
            return y

        def prior_fns():
            return prior.sample, prior.log_prob

        def simulator_fn(config):
            return _simulate

        def unnormalized_posterior():
            def _likelihood_fn(theta, y):
                lp1 = distrax.Independent(
                    distrax.Normal(theta, scales_params[0]), reinterpreted_batch_ndims=1
                ).log_prob(y)
                lp2 = distrax.Independent(
                    distrax.Normal(theta, scales_params[1]), reinterpreted_batch_ndims=1
                ).log_prob(y)
                lp = jnp.logaddexp(jnp.log(0.5) + lp1, jnp.log(0.5) + lp2)
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
    config.experiment_name = "gaussian_mixture"
    config.rng_key = 42
    config.model_fns = model_fn

    config.data = new_dict(
        ndim_data=2,
        ndim_theta=2,
    )

    return config
