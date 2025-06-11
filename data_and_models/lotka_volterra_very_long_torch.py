import ml_collections

from data_and_models import lotka_volterra_torch


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "lotka_volterra_very_long"
    config.rng_key = 42
    config.model_fns = lotka_volterra_torch.model_fn

    config.data = new_dict(
        ndim_data=1000,
        t_end=30,
        ndim_theta=4,
    )

    return config
