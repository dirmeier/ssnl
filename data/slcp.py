from collections import namedtuple
from functools import partial
import numpy as np
import distrax
from jax import numpy as jnp
from jax import random
from jax import scipy as jsp


class slcp_model:
    """
    Model of SNL: Appendix A.1

    Papamakarios et al., 2019
    """

    def __new__(cls, config):
        y_observed = jnp.array(
            [
                [
                    -0.9707123,
                    -2.9461224,
                    -0.4494722,
                    -3.4231849,
                    -0.13285634,
                    -3.364017,
                    -0.85367596,
                    -2.4271638,
                ]
            ]
        )

        def _data_fn(seed):
            y = y_observed
            if "ndim_additional_noise" in config:
                p_noise = distrax.Normal(jnp.zeros(config.ndim_additional_noise), 1.0)
                noise = p_noise.sample(seed=seed, sample_shape=(y.shape[0],))
                y = jnp.hstack([y_observed, noise])
            return y

        def _prior_fns():
            p = distrax.Independent(
                distrax.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0)), 1
            )
            return p.sample, p.log_prob

        def _simulator_fn(seed, theta):
            orig_shape = theta.shape
            if theta.ndim == 2:
                theta = theta[:, None, :]
            us_key, noise_key = random.split(seed)

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
                p_noise = distrax.Normal(jnp.zeros(config.ndim_additional_noise), 1.0)
                noise = p_noise.sample(seed=noise_key, sample_shape=(*y.shape[:-1],))
                y = jnp.concatenate([y, noise], axis=-1)

            return y

        def _unnormalized_posterior_fn():
            def _likelihood_fn(theta, y):
                mu = jnp.tile(theta[:2], 4)
                s1, s2 = theta[2] ** 2, theta[3] ** 2
                corr = s1 * s2 * jnp.tanh(theta[4])
                cov = jnp.array([[s1**2, corr], [corr, s2**2]])
                cov = jsp.linalg.block_diag(*[cov for _ in range(4)])
                p = distrax.MultivariateNormalFullCovariance(mu, cov)
                return p.log_prob(y)

            def log_density_fn(theta, y):
                prior_lp = distrax.Independent(
                    distrax.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0)), 1
                ).log_prob(theta)
                likelihood_lp = _likelihood_fn(theta, y)

                lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
                return lp

            log_density_partial = partial(log_density_fn, y=y_observed)
            log_density = lambda x: log_density_partial(**x)
            return log_density

        def _summary_fn(y):
            if y.ndim == 2:
                y = y[None, ...]
            sumr = jnp.mean(y, axis=1, keepdims=True)
            return sumr

        return (
            _data_fn,
            _prior_fns,
            _simulator_fn,
            _unnormalized_posterior_fn,
            _summary_fn,
        )
