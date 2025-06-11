"""Hyperboloid model.

Adopted from Forbes, et al., Statistics and Computing, 2022.
"""

from functools import partial

import jax
import ml_collections
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.python.internal.backend.jax.gen.linear_operator_lower_triangular import (
    LinearOperatorLowerTriangular,
)
from tensorflow_probability.substrates.jax import distributions as tfd

from data_and_models import data_fn


class model_fn:
    def __new__(cls, config):
        prior = tfd.Independent(
            tfd.Uniform(jnp.full(config.ndim_theta, -2.0), jnp.full(config.ndim_theta, 2.0)), 1
        )
        m11 = jnp.array([-0.5, 0.0])
        m12 = jnp.array([0.5, 0.0])
        m21 = jnp.array([0.0, -0.5])
        m22 = jnp.array([0.0, 0.5])
        nu = 3.0
        scale = jnp.array(0.1)

        def _eudclidean(theta, m1, m2):
            diff = jnp.linalg.norm(theta - m1, ord=2) - jnp.linalg.norm(theta - m2, ord=2)
            return jnp.repeat(diff, config.ndim_data)

        dists_1_fn = jax.vmap(partial(_eudclidean, m1=m11, m2=m12))
        dists_2_fn = jax.vmap(partial(_eudclidean, m1=m21, m2=m22))

        def _simulate(seed, theta):
            mix_key, data_key = jr.split(seed)
            d1 = dists_1_fn(theta).reshape(-1, 1, config.ndim_data)
            d2 = dists_2_fn(theta).reshape(-1, 1, config.ndim_data)
            theta = jnp.concatenate([d1, d2], axis=1)
            idxs = jr.categorical(mix_key, logits=jnp.ones(2), shape=(theta.shape[0],))
            idxs = idxs.reshape(-1, 1, 1)
            locs = jnp.take_along_axis(theta, idxs, 1).squeeze()
            scales = scale * jnp.eye(10)
            distr = tfd.MultivariateStudentTLinearOperator(
                df=nu,
                loc=locs,
                scale=LinearOperatorLowerTriangular(scales),
            )
            y = distr.sample(seed=data_key)
            return y

        def prior_fns():
            return prior.sample, prior.log_prob

        def simulator_fn(config):
            return _simulate

        def unnormalized_posterior():
            def _likelihood_fn(theta, y):
                theta = theta.reshape(-1, config.ndim_theta)
                d1 = dists_1_fn(theta).reshape(-1, config.ndim_data)
                d2 = dists_2_fn(theta).reshape(-1, config.ndim_data)
                scales = scale * jnp.eye(10)
                lp1 = tfd.MultivariateStudentTLinearOperator(
                    df=nu,
                    loc=d1,
                    scale=LinearOperatorLowerTriangular(scales),
                ).log_prob(y)
                lp2 = tfd.MultivariateStudentTLinearOperator(
                    df=nu,
                    loc=d2,
                    scale=LinearOperatorLowerTriangular(scales),
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
    config.experiment_name = "hyperboloid"
    config.rng_key = 42
    config.model_fns = model_fn

    config.data = new_dict(
        ndim_data=10,
        ndim_theta=2,
    )

    return config
