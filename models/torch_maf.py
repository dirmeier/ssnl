from sbi.utils.get_nn_models import (
    posterior_nn,
)


def make_model(config):
    td = posterior_nn(
        model="maf",
        num_transforms=config.n_flow_layers,
        hidden_features=config.hidden_features,
        num_blocks=config.n_hidden_layers,
        use_batch_norm=config.use_batch_norm,
    )
    return td
