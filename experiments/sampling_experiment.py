import os
from functools import partial

import jax
import numpy as np
from absl import logging
from jax import random as jr
from sbijax.mcmc import sample_with_nuts
from sbijax.mcmc.slice import sample_with_slice

from experiments.experiment import Experiment


class SamplingExperiment(Experiment):
    def __init__(self, config, data_config, outdir, run_name):
        super().__init__(config, data_config, outdir, run_name)

        data_fn, prior_fns, _, unnormalized_posterior = data_config.model_fns(data_config.data)
        self._observed = data_fn(jr.PRNGKey(self.data_config.rng_key), self.datafile)
        self.prior_sampler_fn, self.prior_log_density_fn = prior_fns()
        self._unnormalized_posterior = partial(unnormalized_posterior(), y=self._observed)

    def fit(self):
        pass

    def sample(self, checkpoints, **kwargs):
        _, outfile = self._sample()
        return [outfile]

    def _sample(self):
        logging.info(f"sampling from posterior using {self.config.evaluation.sampler.sampler}")
        logging.info(f"using observation {self._observed}")
        len_theta = len(self.prior_sampler_fn(seed=jr.PRNGKey(0)))
        params = dict(self.config.evaluation.sampler)
        params.pop("sampler", None)

        if self.config.evaluation.sampler.sampler == "slice":
            sampling_fn = sample_with_slice

            def lp__(theta):
                return jax.vmap(self._unnormalized_posterior)(theta)

        else:
            sampling_fn = sample_with_nuts

            def lp__(theta):
                return self._unnormalized_posterior(**theta)

        samples = sampling_fn(
            rng_key=jr.PRNGKey(self.config.evaluation.rng_key),
            lp=lp__,
            len_theta=len_theta,
            prior=self.prior_sampler_fn,
            **params,
        )

        samples = samples.reshape(-1, len_theta)
        self._save_pickle(
            self.particles_outfile(),
            {"samples": np.asarray(samples), "observed": np.squeeze(np.asarray(self._observed))},
        )
        return samples, self.particles_outfile()

    def particles_outfile(self):
        outname = self.run_name + "-posteriors.pkl"
        outname = os.path.join(self.outdir, outname)
        return outname
