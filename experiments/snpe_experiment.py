import os
import pickle
from copy import deepcopy

import numpy as np
from absl import logging
from data import dataset
from experiments.experiment import Experiment
from sbi.inference import SNPE
from jax import random


class SNPEExperiment(Experiment):
    def __init__(self, FLAGS, run, run_name):
        super().__init__(FLAGS, run, run_name)

        data_fn, prior_fns, simulator_fn, _, _ = dataset.load_dataset(
            self.config.rng_seq_key * self.config.data.rng_key, self.config
        )

        self._observed = data_fn(
            random.PRNGKey(self.config.rng_seq_key * self.config.data.rng_key)
        )
        self._simulator_fn = simulator_fn
        self._prior = prior_fns()

    def _get_model(self):
        from models import torch_maf

        model = torch_maf.make_model(self.config.model)
        return model

    def get_model_and_params(self):
        logging.info(f"initializing model {self.config.model_type}")
        model = self._get_model()
        return model

    def instance(self):
        model = self.get_model_and_params()
        snpe = SNPE(self._prior, density_estimator=model)
        return snpe

    def fit(self):
        logging.info(f"doing snpe for n_rounds={self.config.training.n_rounds}")
        proposal = self._prior
        model = self.instance()
        key = random.PRNGKey(self.config.rng_seq_key * self.config.training.rng_seq_key)

        all_posteriors = []
        for i in range(self.config.training.n_rounds):
            rng_key, key = random.split(key)
            theta = proposal.sample((self.config.training.n_simulations_per_round,))            
            x = self._simulator_fn(rng_key, theta)
            _ = model.append_simulations(
                theta,
                x,
                proposal=proposal,
            ).train(
                training_batch_size=self.config.training.batch_size,
                learning_rate=self.config.optimizer.params.learning_rate,
                validation_fraction=self.config.training.percentage_data_as_validation_set,
                max_num_epochs=self.config.training.max_n_iter,
                stop_after_epochs=10,
                retrain_from_scratch=True
            )
            posterior = model.build_posterior(
                sample_with=self.config.training.sampler.get('sampler', 'rejection'),
                mcmc_method = "slice",
                mcmc_parameters={
                    "thin": self.config.training.sampler.n_thin
                }
            ).set_default_x(self._observed)

            proposal = posterior
            all_posteriors.append(deepcopy(posterior))

        self.save(all_posteriors)
        return all_posteriors

    def save(self, posterior, **kwargs):
        d = {
            "params": posterior,
            "config": self.config,
            "filename": self.file_name(),
            **kwargs,
        }
        with open(self.training_outfile(), "wb") as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate(self, checkpoint, **kwargs):
        posterior = self._load(checkpoint)
        return self._sample(posterior, **kwargs)

    def _sample_one(self, i, posterior):
        logging.info(f"sampling with parameter set from iteration {i}")
        snpe_samples = posterior.sample(
            sample_shape=(self.config.evaluation.n_samples,),
            x=self._observed,
        )

        n_simul = (i + 1) * self.config.training.n_simulations_per_round
        with open(self.particles_out_name(i), "wb") as handle:
            pickle.dump(
                {
                    "samples": np.asarray(snpe_samples),
                    "n_total_simulations": n_simul,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return snpe_samples

    def _sample(self, all_posteriors, **kwargs):
        if "round" in kwargs:
            round = kwargs["round"]
            self._sample_one(round - 1, all_posteriors[round - 1])
        else:
            for i, posterior in enumerate(all_posteriors):
                self._sample_one(i, posterior)

    def particles_out_name(self, i):
        outname = self.run_name + f"-n_rounds={i + 1}-posteriors.pkl"
        outname = os.path.join(self.FLAGS.outdir, outname)
        return outname

    def file_suffix(self):
        n_rounds = self.config.training.n_rounds
        return self.run_name + f"-n_rounds={n_rounds}"

    def file_name(self):
        return os.path.join(self.FLAGS.outdir, self.file_suffix())
