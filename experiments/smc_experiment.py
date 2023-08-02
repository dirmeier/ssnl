import os
import pickle

import haiku as hk
import jax
from absl import logging
from data import dataset
from experiments.experiment import Experiment
from jax import numpy as jnp, random
from sbijax import SMCABC


class SMCExperiment(Experiment):
    def __init__(self, FLAGS, run, run_name):
        super().__init__(FLAGS, run, run_name)

        (
            data_fn,
            prior_fns,
            simulator_fn,
            unnormalized_posterior_fn,
            summary_fn,
        ) = dataset.load_dataset(
            self.config.rng_seq_key * self.config.data.rng_key, self.config
        )

        def distance_fn(y_simulated, y_observed):
            diff = y_simulated - y_observed
            dist = jax.vmap(lambda el: jnp.linalg.norm(el))(diff)
            return dist

        self._observed = data_fn(
            random.PRNGKey(self.config.rng_seq_key * self.config.data.rng_key)
        )
        self._simulator_fn = simulator_fn
        self._prior_fns = prior_fns
        self._unnormalized_posterior_fn = unnormalized_posterior_fn
        self._summary_fn = summary_fn
        self._distance_fn = distance_fn

    def instance(self):
        prior_sampler, prior_fn = self._prior_fns()
        fns = (prior_sampler, prior_fn), self._simulator_fn
        smc = SMCABC(fns, self._summary_fn, self._distance_fn)
        return smc

    def fit(self):
        self.save()

    def save(self, **kwargs):
        d = {"config": self.config, "filename": self.file_name(), **kwargs}
        with open(self.training_outfile(), "wb") as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate(self, checkpoint):
        smc = self.instance()        
        smc.observed = jnp.atleast_2d(self._observed)
        smc.summarized_observed = self._summary_fn(smc.observed)
        smc._rng_seq = hk.PRNGSequence(
            self.config.rng_seq_key * self.config.evaluation.rng_seq_key
        )
        return self._sample(smc)

    def _sample(self, smc: SMCABC):
        logging.info(
            f"sampling using smcabc for n_rounds={self.config.evaluation.n_rounds}"
        )
        _, smc_info = smc.sample_posterior(
            n_rounds=self.config.evaluation.n_rounds,
            n_particles=self.config.evaluation.n_particles,
            n_simulations_per_theta=self.config.evaluation.n_simulations_per_theta,
            eps_step=self.config.evaluation.eps_step,
            ess_min=self.config.evaluation.ess_min,
            cov_scale=self.config.evaluation.cov_scale,
        )
        len_theta = len(self._prior_fns()[0](seed=random.PRNGKey(0)))

        all_n_simulations, all_particles = smc_info.n_simulations, smc_info.particles
        for i, samples in enumerate(all_particles):
            all_particles[i] = samples.reshape(-1, len_theta)

        for i, (n_simulations, particles) in enumerate(
            zip(all_n_simulations, all_particles)
        ):
            with open(self.particles_out_name(i), "wb") as handle:
                pickle.dump(
                    {"samples": particles, "n_total_simulations": n_simulations},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        return all_particles[-1]

    def particles_out_name(self, i):
        outname = self.run_name + f"-n_rounds={i + 1}-posteriors.pkl"
        outname = os.path.join(self.FLAGS.outdir, outname)
        return outname

    def file_suffix(self):
        n_rounds = self.config.evaluation.n_rounds
        return self.run_name + f"-n_rounds={n_rounds}"

    def file_name(self):
        return os.path.join(self.FLAGS.outdir, self.file_suffix())
