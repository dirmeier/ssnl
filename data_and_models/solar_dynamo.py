"""Solar dynamo model.

Adopted from [1]. Implements Eqn 2 and 3 of [2], and Eqn 13 of [1].

[1] Albert, Carlo, et al. "Learning summary statistics for Bayesian inference with autoencoders."
    SciPost Physics Core, 2022.
[2] Charbonneau, Paul, et al. "Fluctuations in Babcock-Leighton dynamos. II. Revisiting the Gnevyshev-Ohl rule."
    The Astrophysical Journal, 2007.
"""

import distrax
import ml_collections
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.special import erf

from data_and_models import data_fn


def sample_timeseries(seed, y0, alpha_min, alpha_max, epsilon_max, len_timeseries=200):
    a = distrax.Uniform(alpha_min, alpha_max).sample(seed=seed, sample_shape=(len_timeseries,))
    noise = distrax.Uniform(0.0, epsilon_max).sample(seed=seed, sample_shape=(len_timeseries,))

    def _fn(fs, arrays):
        alpha, epsilon = arrays
        f, pn = fs
        f = babcock_leighton_fn(pn)
        pn = babcock_leighton(pn, alpha, epsilon)
        return (f, pn), (f, pn)

    _, (f, y) = lax.scan(_fn, (y0, y0), (a, noise))
    return f.T, y.T, a.T, noise.T


def babcock_leighton_fn(p, b_1=0.6, w_1=0.2, b_2=1.0, w_2=0.8):
    f = 0.5 * (1.0 + erf((p - b_1) / w_1)) * (1.0 - erf((p - b_2) / w_2))
    return f


def babcock_leighton(p, alpha, epsilon):
    p = alpha * babcock_leighton_fn(p) * p + epsilon
    return p


class model_fn:
    def __new__(cls, config):
        prior = distrax.Independent(
            distrax.Uniform(jnp.array([0.9, 0.05, 0.02]), jnp.array([1.4, 0.25, 0.15])),
            1,
        )

        def prior_fns():
            return prior.sample, prior.log_prob

        def _simulate(seed, theta, len_timeseries=100):
            orig_shape = theta.shape
            if theta.ndim == 2:
                theta = theta[None, :, :]

            alpha_min = theta[..., 0]
            alpha_max = alpha_min + theta[..., 1]
            epsilon_max = theta[..., 2]
            y0 = jnp.ones(theta.shape[:-1])

            _, y, _, _ = sample_timeseries(
                seed, y0, alpha_min, alpha_max, epsilon_max, len_timeseries
            )

            y = jnp.swapaxes(y, 1, 0)
            if len(orig_shape) == 2:
                y = y.reshape((*orig_shape[:1], len_timeseries))
            return y

        def simulator_fn(config):
            return _simulate

        return data_fn(config, _simulate, prior), prior_fns, simulator_fn(config), None


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "solar_dynamo"
    config.rng_key = 42
    config.model_fns = model_fn

    config.data = new_dict(
        ndim_data=100,
        ndim_theta=3,
    )

    return config
