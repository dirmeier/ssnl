"""SLCP model.

Adopted from Papamakarios, et al., AISTATS 2019.
"""

from functools import partial

import distrax
import ml_collections
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from tensorflow_probability.python.internal.backend.jax.gen.linear_operator_lower_triangular import (
    LinearOperatorLowerTriangular,
)
from tensorflow_probability.substrates.jax import distributions as tfd

from data_and_models import data_fn


class model_fn:
    def __new__(cls, config):
        prior = distrax.Independent(distrax.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0)), 1)
        total_dim = config.ndim_data
        if "ndim_additional_noise" in config:
            ndim_noise = config.ndim_additional_noise
            total_dim += ndim_noise
            n_mix = 20
            scales = []
            means = jr.normal(jr.PRNGKey(1), (n_mix, config.ndim_additional_noise))
            for i in range(n_mix):
                num = (ndim_noise * (ndim_noise - 1)) // 2
                offdiag = jr.normal(jr.PRNGKey(i), shape=(num,))
                scale = jnp.diag(jnp.exp(jr.normal(jr.PRNGKey(i), (ndim_noise,))))
                scale = scale.at[jnp.tril_indices_from(scale, k=-1)].set(offdiag)
                scales.append(scale.reshape(1, ndim_noise, ndim_noise))
            scales = jnp.vstack(scales)
            p_noise = tfd.MixtureSameFamily(
                tfd.Categorical(logits=jnp.ones(n_mix, dtype=jnp.float32)),
                tfd.MultivariateStudentTLinearOperator(
                    df=2.0, loc=means, scale=LinearOperatorLowerTriangular(scales)
                ),
            )
        idxs = jr.choice(
            jr.PRNGKey(0),
            jnp.arange(total_dim),
            shape=(total_dim,),
            replace=False,
        )

        def _simulate(seed, theta):
            orig_shape = theta.shape
            if theta.ndim == 2:
                theta = theta[:, None, :]
            us_key, noise_key = jr.split(seed)

            def _unpack_params(ps):
                m0 = ps[..., [0]]
                m1 = ps[..., [1]]
                s0 = ps[..., [2]] ** 2
                s1 = ps[..., [3]] ** 2
                r = np.tanh(ps[..., [4]])
                return m0, m1, s0, s1, r

            m0, m1, s0, s1, r = _unpack_params(theta)
            us = distrax.Normal(0.0, 1.0).sample(
                seed=us_key, sample_shape=(theta.shape[0], theta.shape[1], 4, 2)
            )
            xs = jnp.empty_like(us)
            xs = xs.at[:, :, :, 0].set(s0 * us[:, :, :, 0] + m0)
            y = xs.at[:, :, :, 1].set(
                s1 * (r * us[:, :, :, 0] + np.sqrt(1.0 - r**2) * us[:, :, :, 1]) + m1
            )

            if len(orig_shape) == 2:
                y = y.reshape((*theta.shape[:1], 8))
            else:
                y = y.reshape((*theta.shape[:2], 8))

            if "ndim_additional_noise" in config:
                noise = p_noise.sample(seed=noise_key, sample_shape=(*y.shape[:-1],))
                y = jnp.concatenate([y, noise], axis=-1)
                y = y[:, idxs]
            return y

        def prior_fns():
            return prior.sample, prior.log_prob

        def simulator_fn(config):
            fn = partial(_simulate, config=config)
            return fn

        def unnormalized_posterior():
            def _likelihood_fn(theta, y):
                mu = jnp.tile(theta[:2], 4)
                s1, s2 = theta[2] ** 2, theta[3] ** 2
                corr = s1 * s2 * jnp.tanh(theta[4])
                cov = jnp.array([[s1**2, corr], [corr, s2**2]])
                cov = jsp.linalg.block_diag(*[cov for _ in range(4)])
                p = distrax.MultivariateNormalFullCovariance(mu, cov)
                return p.log_prob(y)

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
    config.experiment_name = "slcp"
    config.rng_key = 42
    config.model_fns = model_fn

    config.data = new_dict(
        ndim_data=8,
        ndim_theta=5,
    )

    return config
