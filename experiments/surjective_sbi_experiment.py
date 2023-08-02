import os
import pickle

import haiku as hk
import optax
from absl import logging
from data import dataset
from experiments.experiment import Experiment
from jax import numpy as jnp, random
from sbijax import SNL


class SurjectiveSBIExperiment(Experiment):
    def __init__(self, FLAGS, run, run_name):
        super().__init__(FLAGS, run, run_name)

        (
            data_fn,
            prior_fns,
            simulator_fn,
            unnormalized_posterior_fn,
            _,
        ) = dataset.load_dataset(
            self.config.rng_seq_key * self.config.data.rng_key, self.config
        )

        self._observed = data_fn(
            random.PRNGKey(self.config.rng_seq_key * self.config.data.rng_key)
        )
        self._simulator_fn = simulator_fn
        self._prior_fns = prior_fns
        self._unnormalized_posterior_fn = unnormalized_posterior_fn

    def _get_model(self):
        from models import inference_surjection

        model = inference_surjection.make_model(self.config.model, self.config.data)
        return model

    def get_model_and_params(self):
        logging.info(f"initializing model {self.config.model_type}")
        model = self._get_model()
        return model

    def instance(self):
        model = self.get_model_and_params()
        prior_sampler, prior_fn = self._prior_fns()
        fns = (prior_sampler, prior_fn), self._simulator_fn
        snl = SNL(fns, model)
        return snl, model

    def fit(self):
        logging.info(f"doing snl for n_rounds={self.config.training.n_rounds}")
        optimizer = self.get_optimizer()
        snl, model = self.instance()
        _, info = snl.fit(
            rng_key=self.config.rng_seq_key * self.config.training.rng_seq_key,
            observed=self._observed,
            optimizer=optimizer,
            n_rounds=self.config.training.n_rounds,
            n_simulations_per_round=self.config.training.n_simulations_per_round,
            max_n_iter=self.config.training.max_n_iter,
            batch_size=self.config.training.batch_size,
            percentage_data_as_validation_set=self.config.training.percentage_data_as_validation_set,
            **self.config.training.sampler,
        )

        ess, rhats, all_params = [], [], []
        for diag in info.diagnostics[1:]:
            ess.append(diag.ess)
            rhats.append(diag.rhat)

        all_params = info.params

        info = {"rhats": rhats, "ess": ess, "losses": info.losses}
        self.save(model, all_params, info=info)
        return model, all_params, info

    def save(self, model, params, **kwargs):
        d = {
            "params": params,
            "config": self.config,
            "filename": self.file_name(),
            **kwargs,
        }
        try:
            with open(self.training_outfile(), "wb") as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logging.error(str(e))
            d = {"params": params, "config": self.config, "filename": self.file_name()}
            with open(self.training_outfile(), "wb") as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate(self, checkpoint, **kwargs):
        params = self._load(checkpoint)
        snl, model = self.instance()        
        snl.observed = jnp.atleast_2d(self._observed)
        snl._rng_seq = hk.PRNGSequence(
            self.config.rng_seq_key * self.config.evaluation.rng_seq_key
        )
        self._sample(params, snl, **kwargs)

    def _sample(self, all_params, snl, **kwargs):
        if "round" in kwargs:
            round = kwargs["round"]
            self._sample_one(round - 1, all_params[round - 1], snl)
        else:
            for i, params in enumerate(all_params):
                self._sample_one(i, params, snl)

    def _sample_one(self, i, params, snl):
        logging.info(f"sampling with parameter set from iteration {i}")
        snl_samples, diagnostics = snl.sample_posterior(
            params, **self.config.evaluation.sampler
        )
        len_theta = len(self._prior_fns()[0](seed=random.PRNGKey(0)))
        snl_samples = snl_samples.reshape(-1, len_theta)
        n_simul = (i + 1) * self.config.training.n_simulations_per_round

        try:
            with open(self.particles_out_name(i), "wb") as handle:
                pickle.dump(
                    {
                        "samples": snl_samples,
                        "diagnostics": {
                            "ess": diagnostics.ess,
                            "rhat": diagnostics.rhat,
                        },
                        "n_total_simulations": n_simul,
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as e:
            logging.error(str(e))
            with open(self.particles_out_name(i), "wb") as handle:
                pickle.dump(
                    {"samples": snl_samples, "n_total_simulations": n_simul},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        return snl_samples

    def particles_out_name(self, i):
        outname = self.run_name + f"-n_rounds={i + 1}-posteriors.pkl"
        outname = os.path.join(self.FLAGS.outdir, outname)
        return outname

    def file_suffix(self):
        n_rounds = self.config.training.n_rounds
        return self.run_name + f"-n_rounds={n_rounds}"

    def file_name(self):
        return os.path.join(self.FLAGS.outdir, self.file_suffix())

    def get_optimizer(self):
        optimizer = optax.adam(
            learning_rate=self.config.optimizer.params.learning_rate,
            b1=self.config.optimizer.params.b1,
            b2=self.config.optimizer.params.b2,
            eps=self.config.optimizer.params.eps,
        )
        return optimizer
