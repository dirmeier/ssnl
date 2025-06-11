"""SIR model.

Adopted from Lueckmann, et al., AISTATS, 2021.
"""


import ml_collections
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from data_and_models import data_fn


class model_fn:
    def __new__(cls, config):
        prior = tfd.Independent(
            tfd.LogNormal(jnp.log(jnp.array([0.4, 1.0 / 8.0])), jnp.array([0.5, 0.2])), 1
        )
        N = 1_000_000
        y0 = (N - 1, 1, 0)

        def _solve_ode(theta):
            def f(t, y, args):
                S, I, R = y
                beta, gamma = args
                d_s = -beta * S * I / N
                d_i = beta * S * I / N - gamma * I
                d_r = gamma * I
                d_y = d_s, d_i, d_r
                return d_y

            term = ODETerm(f)
            solver = Tsit5()
            t0 = 0
            t1 = config.t_end
            dt0 = 0.1
            saveat = SaveAt(ts=jnp.linspace(t0, t1, config.ndim_data))

            locs = jnp.zeros((theta.shape[0], config.ndim_data))
            for i in range(theta.shape[0]):
                args = tuple(theta[i])
                sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
                locs = locs.at[i].set(sol.ys[1])
            locs = jnp.nan_to_num(locs)
            return locs

        def _simulate(seed, theta):
            data_key, dequant_key = jr.split(seed)
            locs = _solve_ode(theta)
            ys = tfd.Binomial(total_count=1_000, probs=locs / N).sample(seed=data_key)
            ys = ys + jr.uniform(dequant_key, ys.shape)
            return ys

        def prior_fns():
            return prior.sample, prior.log_prob

        def simulator_fn(config):
            return _simulate

        def unnormalized_posterior():
            def _likelihood_fn(theta, y):
                theta = theta.reshape(-1, config.ndim_theta)
                locs = _solve_ode(theta)
                distr = tfd.Independent(tfd.Binomial(1_000, locs / N), 1)
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
    config.experiment_name = "sir"
    config.rng_key = 42
    config.model_fns = model_fn

    config.data = new_dict(
        ndim_data=100,
        t_end=160,
        ndim_theta=2,
    )

    return config
