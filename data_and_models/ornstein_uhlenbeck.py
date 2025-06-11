"""Ornstein Uhlenbeck model.

Adopted from Särkkä and Solin, Cambridge University Press, 2019.
"""


import ml_collections
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from data_and_models import data_fn


class model_fn:
    def __new__(cls, config):
        prior = tfd.Independent(tfd.Uniform(0.0, jnp.array([10.0, 5.0, 2.0])), 1)
        y0 = 0.0
        T_end = config.t_end
        ts = jnp.linspace(0.0, T_end, config.ndim_data)

        def _mean_scale(theta):
            alpha, beta, sigma = theta[:, [0]], theta[:, [1]], theta[:, [2]]
            mean = alpha + (y0 - alpha) * jnp.exp(-beta * ts)
            var = jnp.square(sigma) / (2 * beta) * (1.0 - jnp.exp(-2 * beta * ts))
            scale = jnp.sqrt(var)
            return mean, scale

        def _simulate(seed, theta):
            mean, scale = _mean_scale(theta)
            ys = tfd.Normal(mean, scale).sample(seed=seed)
            return ys

        def prior_fns():
            return prior.sample, prior.log_prob

        def simulator_fn(config):
            return _simulate

        def unnormalized_posterior():
            def _likelihood_fn(theta, y):
                theta = theta.reshape(-1, config.ndim_theta)
                mean, scale = _mean_scale(theta)
                distr = tfd.Independent(tfd.Normal(mean, scale), 1)
                lp = distr.log_prob(y)
                return lp

            def log_density_fn(theta, y):
                prior_lp = prior.log_prob(theta)
                likelihood_lp = _likelihood_fn(theta, y)

                lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
                return lp

            return log_density_fn

        return (data_fn(config), prior_fns, simulator_fn(config), unnormalized_posterior)


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
