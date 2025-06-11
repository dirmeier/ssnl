import jax
import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.method_name = "snl"
    config.rng_key = 23

    config.model = new_dict(
        n_flow_layers=[
            new_dict(
                type="bijection",
                conditioner=new_dict(type="mlp", ndim_hidden_layers=50, n_hidden_layers=2),
            ),
            new_dict(
                type="bijection",
                conditioner=new_dict(type="mlp", ndim_hidden_layers=50, n_hidden_layers=2),
            ),
            new_dict(
                type="bijection",
                conditioner=new_dict(type="mlp", ndim_hidden_layers=50, n_hidden_layers=2),
            ),
            new_dict(
                type="bijection",
                conditioner=new_dict(type="mlp", ndim_hidden_layers=50, n_hidden_layers=2),
            ),
            new_dict(
                type="bijection",
                conditioner=new_dict(type="mlp", ndim_hidden_layers=50, n_hidden_layers=2),
            ),
        ],
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
            sampler="nuts",
            n_samples=10000,
            n_warmup=5000,
            n_chains=4,
        ),
    )

    config.evaluation = new_dict(
        rng_key=12345,
        sampler=new_dict(
            sampler="nuts",
            n_samples=10000,
            n_warmup=5000,
            n_chains=4,
        ),
    )

    config.optimizer = new_dict(
        name="adam",
        params=new_dict(
            learning_rate=1e-4,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        ),
    )

    return config
