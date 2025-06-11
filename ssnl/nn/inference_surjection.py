import distrax
import haiku as hk
import jax
import numpy as np
from absl import logging
from jax import numpy as jnp
from surjectors import Chain, MaskedAutoregressive, Permutation, TransformedDistribution
from surjectors.conditioners import MADE, mlp_conditioner, transformer_conditioner
from surjectors.surjectors.affine_masked_autoregressive_inference_funnel import (
    AffineMaskedAutoregressiveInferenceFunnel,
)
from surjectors.util import unstack


def _bijector_fn(params):
    means, log_scales = unstack(params, -1)
    return distrax.ScalarAffine(means, jnp.exp(log_scales))


def _base_distribution_fn(n_dim):
    base_distribution = distrax.Independent(
        distrax.Normal(jnp.zeros(n_dim), jnp.ones(n_dim)),
        reinterpreted_batch_ndims=1,
    )
    return base_distribution


def _conditioner_fn(conditioner, output_dim):
    if conditioner.type == "mlp":
        return mlp_conditioner(
            [conditioner.ndim_hidden_layers] * conditioner.n_hidden_layers + [output_dim],
        )
    elif conditioner.type == "transformer":
        return transformer_conditioner(
            output_dim,
            conditioner.num_heads,
            conditioner.num_layers,
            conditioner.key_size,
            conditioner.dropout_rate,
            conditioner.widening_factor,
        )
    logging.fatal("didnt find correct conditioner type")
    raise ValueError("didnt find correct conditioner type")


def _decoder_fn(decoder, n_dimension):
    if decoder.type == "mlp":
        decoder_net = mlp_conditioner(
            [decoder.ndim_hidden_layers] * decoder.n_hidden_layers + [n_dimension * 2],
            activation=jax.nn.tanh,
        )
    else:
        logging.fatal("didnt find correct decoder type")
        raise ValueError("didnt find correct decoder type")

    def _fn(z):
        params = decoder_net(z)
        mu, log_scale = jnp.split(params, 2, -1)
        return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)), 1)

    return _fn


def make_model(config, data_config):
    def _flow(method, **kwargs):
        dim = data_config.ndim_data
        if "ndim_additional_noise" in data_config:
            dim += data_config.ndim_additional_noise

        layers = []
        order = jnp.arange(dim)
        for i, entry in enumerate(config.n_flow_layers):
            if entry.type == "bijection":
                layer = MaskedAutoregressive(
                    bijector_fn=_bijector_fn,
                    conditioner=MADE(
                        input_size=dim,
                        hidden_layer_sizes=[entry.conditioner.ndim_hidden_layers]
                        * entry.conditioner.n_hidden_layers,
                        n_params=2,
                        w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                        b_init=jnp.zeros,
                        activation=jax.nn.tanh,
                    ),
                )
                layers.append(layer)
                order = order[::-1]
                layers.append(Permutation(order, 1))
            elif entry.type == "funnel":
                reduc = entry["reduction_factor"]
                reduc = config.reduction_factor if np.isnan(reduc) else reduc
                latent_dim = int(reduc * dim)
                layer = AffineMaskedAutoregressiveInferenceFunnel(
                    n_keep=latent_dim,
                    decoder=_decoder_fn(entry.decoder, dim - latent_dim),
                    conditioner=MADE(
                        input_size=latent_dim,
                        hidden_layer_sizes=[entry.conditioner.ndim_hidden_layers]
                        * entry.conditioner.n_hidden_layers,
                        n_params=2,
                        w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                        b_init=jnp.zeros,
                        activation=jax.nn.tanh,
                    ),
                )
                layers.append(layer)
                dim = latent_dim
                order = order[::-1]
                order = order[:dim]
                order = order - jnp.min(order)
                layers.append(Permutation(order, 1))
            else:
                logging.fatal("didnt find correct flow type")
                raise ValueError("didnt find correct flow type")
        layers = layers[:-1]
        chain = Chain(layers)

        td = TransformedDistribution(_base_distribution_fn(dim), chain)
        return td(method, **kwargs)

    td = hk.transform(_flow)
    td = hk.without_apply_rng(td)
    return td
