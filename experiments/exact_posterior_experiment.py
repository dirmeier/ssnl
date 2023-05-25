import os
import pickle

import haiku as hk
from data import dataset
from experiments.experiment import Experiment
from jax import random
from sbijax.mcmc import sample_with_nuts
from absl import logging
from sbijax.mcmc.slice import sample_with_slice


class ExactPosteriorExperiment(Experiment):
    def __init__(self, FLAGS, run, run_name):
        super().__init__(FLAGS, run, run_name)
        data_fn, prior_fns, _, unnormalized_posterior_fn, _ = dataset.load_dataset(
            self.config.data.rng_key, self.config
        )
        self._prior_fns = prior_fns
        self.prior_sampler_fn, self.prior_log_density_fn = prior_fns()
        self._observed = data_fn(self.config.data.rng_key)
        self._unnormalized_posterior_fn = unnormalized_posterior_fn()

    def fit(self):
        pass

    def save(self, **kwargs):
        d = {"config": self.config, "filename": self.file_name(), **kwargs}
        with open(self.training_outfile(), "wb") as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate(self, checkpoint):
        return self._sample()

    def _sample(self):
        logging.info(
            f"sampling from posterior using {self.config.evaluation.sampler.sampler}"
        )
        len_theta = len(self._prior_fns()[0](seed=random.PRNGKey(0)))
        params = dict(self.config.evaluation.sampler)
        params.pop("sampler", None)

        if self.config.evaluation.sampler.sampler is "slice":
            samples = sample_with_slice(
                rng_seq=hk.PRNGSequence(self.config.evaluation.rng_seq_key),
                lp=self._unnormalized_posterior_fn,
                prior=self.prior_sampler_fn,
                **params,
            )
        else:
            samples = sample_with_nuts(
                rng_seq=hk.PRNGSequence(self.config.evaluation.rng_seq_key),
                lp=self._unnormalized_posterior_fn,
                len_theta=len_theta,
                **params,
            )

        samples = samples.reshape(-1, len_theta)
        with open(self.posteriors_outfile(), "wb") as handle:
            pickle.dump({"samples": samples}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return samples

    def file_suffix(self):
        return self.run_name + "-exact"
