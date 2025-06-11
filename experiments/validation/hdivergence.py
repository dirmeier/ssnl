import numpy as np
from jax import numpy as jnp
from jax import random as jr
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def _kde(x):
    params = {"bandwidth": np.linspace(0.1, 5, 30)}
    grid = GridSearchCV(KernelDensity(), params)
    _ = grid.fit(x)
    return grid


def h_divergence(x, y):
    x, y = np.asarray(x), np.asarray(y)
    x_model, y_model = _kde(x), _kde(y)
    x_prob = -np.mean(x_model.score_samples(x))
    y_prob = -np.mean(y_model.score_samples(y))

    hs = np.zeros((5, 2))
    for i in range(5):
        x_key, y_key = jr.split(jr.PRNGKey(i))
        x_idxs = jr.choice(x_key, y.shape[0], (x.shape[0] // 2,), replace=False)
        y_idxs = jr.choice(y_key, y.shape[0], (y.shape[0] // 2,), replace=False)

        xy = np.asarray(jnp.concatenate([x[x_idxs], y[y_idxs]], axis=0))
        xy_model = _kde(xy)
        mixed = 0.5 * -np.mean(xy_model.score_samples(x)) + 0.5 * -np.mean(
            xy_model.score_samples(y)
        )

        hjs = mixed - 0.5 * (x_prob + y_prob)
        hmin = mixed - min(x_prob, y_prob)
        hs[i] = np.array([hjs, hmin])
    hs = np.mean(hs, axis=0)
    return hs[0], hs[1]
