import os
import pathlib
import pickle

import distrax
from jax import lax
from jax import numpy as jnp
from jax.scipy.special import erf


def sample_timeseries(seed, y0, alpha_min, alpha_max, epsilon_max, len_timeseries=200):
    a = distrax.Uniform(alpha_min, alpha_max).sample(
        seed=seed, sample_shape=(len_timeseries,)
    )
    noise = distrax.Uniform(0.0, epsilon_max).sample(
        seed=seed, sample_shape=(len_timeseries,)
    )

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


class solar_dynamo_model:
    """Implements Eqn 2 and 3 of
    FLUCTUATIONS IN BABCOCK-LEIGHTON DYNAMOS. II. REVISITING THE GNEVYSHEV-OHL RULE
    https://iopscience.iop.org/article/10.1086/511177/pdf

    and Eqn 13 of https://arxiv.org/abs/2201.12059
    """

    def __new__(cls, config):
        def _data_fn(seed):
            """
            Data has been created using (seed=42):

            y0 = 1.0
            alpha1 = 1.11
            alpha2 = alpha1 + 0.15
            epsilon_max = 0.08

            f, y, alpha, noise = sample_timeseries(
                seed, y0, alpha1, alpha2, epsilon_max, 200
            )
            return f, y, alpha, noise
            """

            dir = pathlib.Path(__file__).parent.resolve()
            fl = os.path.join(dir, "solar_dynamo_observation.pkl")
            with open(fl, "rb") as handle:
                data = pickle.load(handle)
                return data["y"]

        def _prior_fns():
            p = distrax.Independent(
                distrax.Uniform(
                    jnp.array([0.9, 0.05, 0.02]), jnp.array([1.4, 0.25, 0.15])
                ),
                1,
            )
            return p.sample, p.log_prob

        def _simulator_fn(seed, theta, len_timeseries=200):
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

        return _data_fn, _prior_fns, _simulator_fn, None, None
