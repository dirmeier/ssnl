from sbi.utils.get_nn_models import posterior_nn


def make_model(config):
    td = posterior_nn(
        model="maf",
        num_transforms=config.n_flow_layers,
        hidden_features=config.hidden_features,
        num_blocks=config.n_hidden_layers,
        use_batch_norm=config.use_batch_norm,
        z_score_x="none",
        z_score_theta="none",
    )
    return td
