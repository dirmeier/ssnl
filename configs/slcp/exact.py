import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "slcp"
    config.dataset = "slcp"
    config.model_type = "nuts"
    config.rng_seq_key = 23

    config.data = new_dict(rng_key=42, ndim_data=8)
    config.training = new_dict()

    config.evaluation = new_dict(
        rng_seq_key=123,
        sampler=new_dict(
            name="slice",
            n_thin=1,
            n_doublings=5,
            n_samples=10000,
            n_warmup=5000,
            n_chains=4,
        ),
    )
    return config
