import numpy as np
import torch
from jax import numpy as jnp
from jax import random


class slcp_model:
    """
    Model of SNL: Appendix A.1

    Papamakarios et al., 2019
    """

    def __new__(cls, config):
        y_observed = torch.asarray(
            [
                -0.9707123,
                -2.9461224,
                -0.4494722,
                -3.4231849,
                -0.13285634,
                -3.364017,
                -0.85367596,
                -2.4271638,
            ],
            dtype=torch.float32,
        )

        def _data_fn(seed):
            y = jnp.asarray(np.asarray(y_observed))
            if "ndim_additional_noise" in config:
                noise = random.normal(seed, shape=(config.ndim_additional_noise,))
                y = jnp.concatenate([y, noise])
            y = np.asarray(y)
            return torch.asarray(y, dtype=torch.float32)

        def _prior_fns():
            p = torch.distributions.Independent(
                torch.distributions.Uniform(
                    low=torch.as_tensor(-3.0 * torch.ones(5), dtype=torch.float32),
                    high=torch.as_tensor(3.0 * torch.ones(5), dtype=torch.float32),
                ),
                1,
            )
            return p

        def _simulator_fn(seed, theta):
            theta = jnp.asarray(np.asarray(theta.cpu()))
            us_key, noise_key = random.split(seed)

            def _unpack_params(ps):
                m0 = ps[..., [0]]
                m1 = ps[..., [1]]
                s0 = ps[..., [2]] ** 2
                s1 = ps[..., [3]] ** 2
                r = np.tanh(ps[..., [4]])
                return m0, m1, s0, s1, r

            m0, m1, s0, s1, r = _unpack_params(theta)
            us = random.normal(us_key, shape=(theta.shape[0], 4, 2))
            xs = jnp.empty_like(us)
            xs = xs.at[:, :, 0].set(s0 * us[:, :, 0] + m0)
            y = xs.at[:, :, 1].set(
                s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r**2) * us[:, :, 1]) + m1
            )
            y = y.reshape(-1, 8)
            y = np.asarray(y)

            if "ndim_additional_noise" in config:
                noise = random.normal(
                    noise_key,
                    shape=(
                        y.shape[0],
                        config.ndim_additional_noise,
                    ),
                )
                y = jnp.concatenate([y, noise], axis=-1)
                y = np.asarray(y)

            y = torch.asarray(y, dtype=torch.float32)
            return y

        return _data_fn, _prior_fns, _simulator_fn, None, None
