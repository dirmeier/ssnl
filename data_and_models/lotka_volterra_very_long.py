import ml_collections
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from data_and_models import lotka_volterra


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "lotka_volterra_very_long"
    config.rng_key = 42
    config.model_fns = lotka_volterra.model_fn

    config.data = new_dict(
        ndim_data=1000,
        t_end=30,
        ndim_theta=4,
    )

    return config
