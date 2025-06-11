import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.method_name = "mcmc"
    config.rng_key = 23
    config.training = new_dict()

    config.evaluation = new_dict(
        rng_key=12345,
        sampler=new_dict(
            sampler="slice",
            n_thin=1,
            n_doubling=5,
            n_samples=20000,
            n_warmup=10000,
            n_chains=10,
        ),
    )
    return config
