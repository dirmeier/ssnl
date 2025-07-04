from functools import partial

import haiku as hk
import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from sbijax import SNL
from sbijax.nn.early_stopping import EarlyStopping

from ssnl.snass.dataloader import DataLoader


def _sample_unit_sphere(rng_key, n, dim):
    u = jr.normal(rng_key, (n, dim))
    norm = jnp.linalg.norm(u, ord=2, axis=-1, keepdims=True)
    return u / norm


def _jsd_summary_loss(params, rng_key, apply_fn, **batch):
    y, theta = batch["y"], batch["theta"]
    n, p = theta.shape

    phi_key, rng_key = jr.split(rng_key)
    summr = apply_fn(params, method="summary", y=y)
    summr = jnp.tile(summr, [10, 1])
    theta = jnp.tile(theta, [10, 1])

    phi = _sample_unit_sphere(phi_key, 10, p)
    phi = jnp.repeat(phi, n, axis=0)

    second_summr = apply_fn(params, method="secondary_summary", y=summr, theta=phi)
    theta_prime = jnp.sum(theta * phi, axis=1).reshape(-1, 1)

    idx_pos = jnp.tile(jnp.arange(n), 10)
    perm_key, rng_key = jr.split(rng_key)
    idx_neg = jax.vmap(lambda x: jr.permutation(x, n))(jr.split(perm_key, 10)).reshape(-1)
    f_pos = apply_fn(params, method="critic", y=second_summr, theta=theta_prime)
    f_neg = apply_fn(
        params,
        method="critic",
        y=second_summr[idx_pos],
        theta=theta_prime[idx_neg],
    )
    a, b = -jax.nn.softplus(-f_pos), jax.nn.softplus(f_neg)
    mi = a.mean() - b.mean()
    return -mi


class _SNASSSNets:
    def __init__(self, dim, second_dim, hidden_size):
        self.dim = dim
        self.secondary_dim = second_dim
        self._summary = hk.nets.MLP(output_sizes=[hidden_size, dim], activation=jax.nn.tanh)
        self._secondary_summary = hk.nets.MLP(
            output_sizes=[hidden_size, self.secondary_dim], activation=jax.nn.tanh
        )
        self._critic = hk.nets.MLP(output_sizes=[hidden_size, 1], activation=jax.nn.tanh)

    def __call__(self, method, **kwargs):
        return getattr(self, method)(**kwargs)

    def forward(self, y, theta):
        s = self.summary(y)
        s2 = self.secondary_summary(s, theta)
        c = self.critic(y[:, : self.secondary_dim], y[:, [0]])
        return s, s2, c

    def summary(self, y):
        return self._summary(y)

    def secondary_summary(self, y, theta):
        return self._secondary_summary(jnp.concatenate([y, theta], axis=-1))

    def critic(self, y, theta):
        return self._critic(jnp.concatenate([y, theta], axis=-1))


def _make_critic(summary_dim, sec_summary_dim, hidden_size):
    @hk.without_apply_rng
    @hk.transform
    def _net(method, **kwargs):
        net = _SNASSSNets(summary_dim, sec_summary_dim, hidden_size)
        return net(method, **kwargs)

    return _net


# pylint: disable=too-many-arguments,unused-argument
class SNASSS(SNL):
    """Sequential neural approximate slice sufficient statistics.

    References:
        .. [1] Yanzhi Chen et al. "Is Learning Summary Statistics Necessary for
           Likelihood-free Inference". ICML, 2023
    """

    def __init__(self, model_fns, density_estimator, summary_dim, sec_summary_dim, hidden_size):
        super().__init__(model_fns, density_estimator)
        self.sc_net = _make_critic(summary_dim, sec_summary_dim, hidden_size)

    # pylint: disable=arguments-differ,too-many-locals
    def fit(
        self,
        rng_key,
        data,
        optimizer=optax.adam(0.0003),
        n_iter=1000,
        batch_size=128,
        percentage_data_as_validation_set=0.1,
        n_early_stopping_patience=10,
        **kwargs,
    ):
        itr_key, rng_key = jr.split(rng_key)
        train_iter, val_iter = self.as_iterators(
            itr_key, data, batch_size, percentage_data_as_validation_set
        )

        snet_params, snet_losses = self._fit_summary_net(
            rng_key=rng_key,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=optimizer,
            n_iter=n_iter,
            n_early_stopping_patience=n_early_stopping_patience,
        )

        train_iter = self._as_summary(train_iter, snet_params)
        val_iter = self._as_summary(val_iter, snet_params)

        nde_params, losses = self._fit_model_single_round(
            seed=rng_key,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=optimizer,
            n_iter=n_iter,
            n_early_stopping_patience=n_early_stopping_patience,
        )

        return {"params": nde_params, "s_params": snet_params}, losses

    def _as_summary(self, iter, params):
        @jax.jit
        def as_batch(y, theta):
            return {
                "y": self.sc_net.apply(params, method="summary", y=y),
                "theta": theta,
            }

        ds = DataLoader(
            num_batches=iter.num_batches,
            batches=[as_batch(**iter(i)) for i in range(iter.num_batches)],
        )
        return ds

    def _fit_summary_net(
        self,
        rng_key,
        train_iter,
        val_iter,
        optimizer,
        n_iter,
        n_early_stopping_patience,
    ):

        init_key, rng_key = jr.split(rng_key)
        params = self._init_summary_net_params(init_key, **train_iter(0))
        state = optimizer.init(params)
        loss_fn = partial(_jsd_summary_loss, apply_fn=self.sc_net.apply)

        @jax.jit
        def step(rng, params, state, **batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, rng, **batch)
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_state

        losses = np.zeros([n_iter, 2])
        early_stop = EarlyStopping(1e-3, n_early_stopping_patience)
        best_params, best_loss = None, np.inf
        logging.info("training summary net")
        for i in range(n_iter):
            train_loss = 0.0
            epoch_key, rng_key = jr.split(rng_key)
            for j in range(train_iter.num_batches):
                batch = train_iter(j)
                batch_loss, params, state = step(jr.fold_in(epoch_key, j), params, state, **batch)
                train_loss += batch_loss * (batch["y"].shape[0] / train_iter.num_samples)
            val_key, rng_key = jr.split(rng_key)
            validation_loss = self._summary_validation_loss(params, val_key, val_iter)
            losses[i] = jnp.array([train_loss, validation_loss])

            _, early_stop = early_stop.update(validation_loss)
            if early_stop.should_stop:
                logging.info("early stopping criterion found")
                break
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_params = params.copy()

        losses = jnp.vstack(losses)[: (i + 1), :]
        return best_params, losses

    def _init_summary_net_params(self, rng_key, **init_data):
        params = self.sc_net.init(rng_key, method="forward", **init_data)
        return params

    def _summary_validation_loss(self, params, rng_key, val_iter):
        loss_fn = jax.jit(partial(_jsd_summary_loss, apply_fn=self.sc_net.apply))

        def body_fn(i, batch_key):
            batch = val_iter(i)
            loss = loss_fn(params, batch_key, **batch)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        losses = 0.0
        for i in range(val_iter.num_batches):
            batch_key, rng_key = jr.split(rng_key)
            losses += body_fn(i, batch_key)
        return losses

    def sample_posterior(
        self,
        rng_key,
        params,
        observable,
        *,
        n_chains=4,
        n_samples=2_000,
        n_warmup=1_000,
        **kwargs,
    ):
        observable = jnp.atleast_2d(observable)
        summary = self.sc_net.apply(params["s_params"], method="summary", y=observable)
        return super().sample_posterior(
            rng_key,
            params["params"],
            summary,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=n_warmup,
            **kwargs,
        )
