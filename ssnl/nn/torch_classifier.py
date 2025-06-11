from typing import Optional

from sbi.neural_nets.classifier import build_input_layer
from sbi.utils.user_input_checks import check_data_device, check_embedding_net_device
from torch import Tensor, nn


def make_model(config):
    def build_fn(batch_theta, batch_x):
        return build_mlp(
            batch_theta,
            batch_x,
            hidden_features=config.hidden_features,
            n_hidden_layers=config.n_hidden_layers,
        )

    return build_fn


def build_mlp(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "none",
    z_score_y: Optional[str] = "none",
    hidden_features: int = 50,
    n_hidden_layers: int = 2,
    embedding_net_x: nn.Module = nn.Identity(),
    embedding_net_y: nn.Module = nn.Identity(),
) -> nn.Module:
    check_data_device(batch_x, batch_y)
    check_embedding_net_device(embedding_net=embedding_net_x, datum=batch_y)
    check_embedding_net_device(embedding_net=embedding_net_y, datum=batch_y)

    # Infer the output dimensionalities of the embedding_net by making a forward pass.
    x_numel = embedding_net_x(batch_x[:1]).numel()
    y_numel = embedding_net_y(batch_y[:1]).numel()

    neural_net = []
    neural_net += [nn.Linear(x_numel + y_numel, hidden_features), nn.ReLU()]
    assert n_hidden_layers >= 1
    for i in range(n_hidden_layers - 1):
        layer = nn.Linear(hidden_features, hidden_features)
        neural_net.append(layer)
        neural_net.append(nn.ReLU())
    neural_net.append(nn.Linear(hidden_features, 1))
    neural_net = nn.Sequential(*neural_net)

    input_layer = build_input_layer(
        batch_x, batch_y, z_score_x, z_score_y, embedding_net_x, embedding_net_y
    )

    neural_net = nn.Sequential(input_layer, neural_net)

    return neural_net
