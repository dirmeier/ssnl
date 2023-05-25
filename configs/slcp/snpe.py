import jax
import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "slcp"
    config.dataset = "slcp"
    config.model_type = "snpe"
    config.rng_seq_key = 23

    config.data = new_dict(rng_key=42, ndim_data=8)

    config.model = new_dict(
        n_flow_layers=5,
        n_hidden_layers=2,
        hidden_features=50,
        use_batch_norm=False,
    )

    config.training = new_dict(
        rng_seq_key=12,
        n_iter=1000,
        batch_size=100,
        n_rounds=1,
        n_simulations_per_round=1000,
        max_n_iter=1000,
        percentage_data_as_validation_set=0.1,
        sampler=new_dict(
            sampler="mcmc",
            n_thin=1,
            n_chains=4,
        ),
    )

    config.evaluation = new_dict(
        rng_seq_key=123,
        n_samples=10000,
    )

    config.optimizer = new_dict(
        name="adam",
        params=new_dict(
            learning_rate=1e-4,
        ),
    )

    return config
