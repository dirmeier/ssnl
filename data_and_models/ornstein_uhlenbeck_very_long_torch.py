import ml_collections

from data_and_models import ornstein_uhlenbeck_torch


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "ornstein_uhlenbeck_very_long"
    config.rng_key = 42
    config.model_fns = ornstein_uhlenbeck_torch.model_fn

    config.data = new_dict(
        ndim_data=1000,
        t_end=10,
        ndim_theta=3,
    )

    return config
