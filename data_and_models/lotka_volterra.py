"""Lotka Volterra model.

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
        prior = tfd.Independent(tfd.LogNormal(jnp.array([-0.125, -3.0, -0.125, -3.0]), 0.5), 1)
        y0 = (30.0, 1.0)

        def _solve_ode(theta):
            def f(t, y, args):
                prey, predator = y
                alpha, beta, gamma, delta = args
                d_prey = alpha * prey - beta * prey * predator
                d_predator = -gamma * predator + delta * prey * predator
                d_y = d_prey, d_predator
                return d_y

            term = ODETerm(f)
            solver = Tsit5()
            t0 = 0
            t1 = config.t_end
            dt0 = 0.1
            saveat = SaveAt(ts=jnp.linspace(t0, t1, config.ndim_data // 2))

            locs = jnp.zeros((theta.shape[0], config.ndim_data))
            for i in range(theta.shape[0]):
                args = tuple(theta[i])
                sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
                locs = locs.at[i].set(jnp.concatenate([sol.ys[0], sol.ys[1]]))
            locs = jnp.nan_to_num(locs)
            locs = locs.clip(1e-03, 10000.0)
            return locs

        def _simulate(seed, theta):
            locs = _solve_ode(theta)
            ys = tfd.LogNormal(jnp.log(locs), 0.1).sample(seed=seed)
            return ys

        def prior_fns():
            return prior.sample, prior.log_prob

        def simulator_fn(config):
            return _simulate

        def unnormalized_posterior():
            def _likelihood_fn(theta, y):
                theta = theta.reshape(-1, config.ndim_theta)
                locs = _solve_ode(theta)
                distr = tfd.Independent(tfd.LogNormal(locs, 0.1), 1)
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
    config.experiment_name = "lotka_volterra"
    config.rng_key = 42
    config.model_fns = model_fn

    config.data = new_dict(
        ndim_data=100,
        t_end=30,
        ndim_theta=4,
    )

    return config
