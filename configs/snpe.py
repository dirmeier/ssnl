import jax
import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.method_name = "snpe"
    config.rng_key = 23

    config.model = new_dict(
        n_flow_layers=5,
        n_hidden_layers=2,
        hidden_features=50,
        use_batch_norm=False,
    )

    config.training = new_dict(
        rng_key=1234,
        n_rounds=5,
        n_iter=2000,
        batch_size=100,
        n_simulations_per_round=1000,
        n_early_stopping_patience=10,
        percentage_data_as_validation_set=0.1,
        sampler=new_dict(
            sampler="mcmc",
            thin=1,
            num_chains=4,
        ),
    )

    config.evaluation = new_dict(
        rng_key=12345,
        n_samples=10000,
        sampler=new_dict(
            sampler="mcmc",
            thin=1,
            num_chains=4,
        ),
    )

    config.optimizer = new_dict(
        name="adam",
        params=new_dict(
            learning_rate=1e-4,
        ),
    )

    return config
