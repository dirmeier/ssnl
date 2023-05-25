import pathlib
import pickle
from os import listdir
from os.path import isfile, join, basename

import click
import re
import numpy as np
import pandas as pd

from experiments.validation import mmd
from jax import random, numpy as jnp


@click.command()
@click.argument("folder", type=str)
@click.argument("exact-posterior", type=str)
@click.argument("outfile", type=str)
@click.argument("n-subsample", type=int, default=10000)
def main(folder, exact_posterior, outfile, n_subsample):
    all_files = lambda path: sorted(
        [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    )

    with open(exact_posterior, "rb") as f:
        exact_posterior = pickle.load(f)

    posteriors = {}
    for fl in all_files(folder):
        if not fl.endswith("posteriors.pkl"):
            continue
        with open(fl, "rb") as f:
            suffix = basename(fl)
            sclp_posterior = pickle.load(f)
            posteriors[suffix] = sclp_posterior

    exact_posterior = exact_posterior["samples"]
    idxs = random.choice(
        random.PRNGKey(0),
        jnp.arange(exact_posterior.shape[0]),
        shape=(jnp.minimum(exact_posterior.shape[0], n_subsample),),
        replace=False,
    )
    exact_posterior = np.asarray(exact_posterior[idxs])

    df = pd.DataFrame()
    key = random.PRNGKey(1)
    for suffix, value in posteriors.items():
        if "exact" in suffix:
            continue
        reg = re.match("sbi-(.*)-(.*)-seed.*(\d+)-n_rounds=(\d+)-posteriors.pkl", suffix)
        rng_key, key = random.split(key)
        samples = np.asarray(value["samples"])
        idxs = random.choice(
            key,
            jnp.arange(samples.shape[0]),
            shape=(jnp.minimum(samples.shape[0], n_subsample),),
            replace=False,
        )
        idxs = np.asarray(idxs)
        sub_sample = samples[idxs]
        print(f"doing {suffix}")
        arr = float(mmd(exact_posterior, sub_sample))

        df = pd.concat([
            df,
            pd.DataFrame({
                "experiment": [reg.group(1)],
                "model": [reg.group(2)],
                "n_rounds": [int(reg.group(4))],
                "mmd": [arr],
                "seed": [reg.group(3)]
            }),
        ], ignore_index=True)

    with open(outfile, "wb") as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
