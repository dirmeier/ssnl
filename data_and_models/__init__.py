import logging
import pathlib
import pickle

import numpy as np
from jax import numpy as jnp
from jax import random as jr


def data_fn(config, simulate, prior):
    def fn(seed, data_filename):
        fl = pathlib.Path(data_filename)
        if fl.exists():
            logging.info(f"reading available data with seed {seed}")
            with open(fl, "rb") as fh:
                data = pickle.load(fh)
            y, theta = data["y"].reshape(1, -1), data["theta"].reshape(1, -1)
        else:
            y_key, theta_key = jr.split(seed)
            theta = prior.sample(seed=theta_key).reshape(1, -1)
            y = simulate(y_key, theta).reshape(1, -1)
            y = np.array(y, copy=True)
            theta = np.array(theta, copy=True)
            with open(fl, "wb") as fh:
                pickle.dump(
                    {
                        "y": y,
                        "theta": theta,
                    },
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        logging.info(f"y: {y}")
        logging.info(f"theta: {theta}")
        return jnp.array(y), jnp.array(theta)

    return fn
