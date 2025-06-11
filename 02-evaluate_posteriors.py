import pickle

import click
import numpy as np
import pandas as pd
import torch
from jax import random as jr
from sbibm.metrics import c2st, mmd

from experiments.validation.hdivergence import h_divergence


@click.command()
@click.argument("approximate-posterior", type=str)
@click.argument("exact-posterior", type=str)
@click.argument("outfile", type=str)
@click.argument("n-subsample", type=int, default=10000)
def main(approximate_posterior, exact_posterior, outfile, n_subsample):
    index = approximate_posterior.split("/")[-1]
    with open(exact_posterior, "rb") as f:
        exact_posterior = pickle.load(f)
    with open(approximate_posterior, "rb") as f:
        approximate_posterior = pickle.load(f)

    exact_posterior,exact_posterior_obs = \
        exact_posterior["samples"], exact_posterior["observed"]
    approximate_posterior, approximate_posterior_obs = \
        approximate_posterior["samples"], approximate_posterior["observed"]

    exact_posterior = jr.choice(
        jr.PRNGKey(23), exact_posterior, (n_subsample,), replace=False
    )
    approximate_posterior = jr.choice(
        jr.PRNGKey(24), approximate_posterior, (n_subsample,), replace=False
    )
    print(exact_posterior_obs)
    print(approximate_posterior_obs)
    if len(exact_posterior_obs) == len(approximate_posterior_obs):
        assert np.allclose(exact_posterior_obs, approximate_posterior_obs, atol=1e-03)
    else:
        assert np.allclose(exact_posterior_obs, approximate_posterior_obs[:len(exact_posterior_obs)], atol=1e-03)

    print("Computing C2ST")
    c2st_accuracy = c2st(
        torch.from_numpy(np.copy(exact_posterior)),
        torch.from_numpy(np.copy(approximate_posterior))
    )
    print("Computing MMD")
    mmd_accuracy = mmd(
        torch.from_numpy(np.copy(exact_posterior)),
        torch.from_numpy(np.copy(approximate_posterior))
    )
    print("Computing H-divergences")
    hjs, hmin = h_divergence(
        np.copy(exact_posterior),
        np.copy(approximate_posterior)
    )
    pd.DataFrame({
        "H-Jenson Shannon": [float(hjs)],
        "H-Min": [float(hmin)],
        "MMD": [float(mmd_accuracy)],
        "C2st": [float(c2st_accuracy)]
    }, index=[index]).to_pickle(outfile)


if __name__ == "__main__":
    main()
