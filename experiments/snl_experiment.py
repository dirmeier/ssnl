import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from sbijax import SNL

from experiments.experiment import Experiment


class SNLExperiment(Experiment):
    def __init__(self, config, data_config, outdir, run_name):
        super().__init__(config, data_config, outdir, run_name)

        data_fn, prior_fns, simulator, _ = data_config.model_fns(data_config.data)
        self._observed = data_fn(jr.PRNGKey(self.data_config.rng_key), self.datafile)
        self._simulator = simulator
        self._prior_fns = prior_fns

    def get_model(self):
        from ssnl.nn import inference_surjection

        logging.info(f"initializing model {self.config.method_name}")
        model = inference_surjection.make_model(self.config.model, self.data_config.data)
        return model

    def instance(self):
        model = self.get_model()
        prior_sampler, prior_fn = self._prior_fns()
        fns = (prior_sampler, prior_fn), self._simulator
        snl = SNL(fns, model)
        return snl, model

    def fit(self):
        logging.info(
            f"doing {self.__class__.__name__} " f"for n_rounds={self.config.training.n_rounds}"
        )
        rng_key = jr.PRNGKey(self.config.rng_key * self.config.training.rng_key)
        data, params = None, {}
        outfiles = []
        for itr in range(self.config.training.n_rounds):
            params, data, outfile = self._fit(jr.fold_in(rng_key, itr), itr + 1, params, data)
            outfiles.append(outfile)
        return outfiles

    def _fit(self, rng_key, rnd, params=None, data=None):
        logging.info(f"doing round={rnd}")
        logging.info(f"using observation {self._observed}")
        optimizer = self.get_optimizer()
        snl, _ = self.instance()

        simulate_key, train_key = jr.split(rng_key)
        new_data, diagnostics = snl.simulate_data(
            simulate_key,
            params=params,
            observable=self._observed,
            data=data,
            n_simulations=self.config.training.n_simulations_per_round,
            **self.config.training.sampler,
        )
        data = snl.stack_data(data, new_data)
        params, losses = snl.fit(
            train_key,
            data=data,
            observed=self._observed,
            optimizer=optimizer,
            n_iter=self.config.training.n_iter,
            batch_size=self.config.training.batch_size,
            percentage_data_as_validation_set=self.config.training.percentage_data_as_validation_set,
            n_early_stopping_patience=self.config.training.n_early_stopping_patience,
        )

        ess = rhat = None
        if diagnostics is not None:
            ess = diagnostics.ess
            rhat = diagnostics.rhat

        dc = {
            "observed": np.squeeze(np.asarray(self._observed)),
            "round": rnd,
            "params": params,
            "losses": losses,
            "new_data": new_data,
            "diagnostics": {"rhat": rhat, "ess": ess},
            "all_data": data,
        }

        self._save_parameters_or_samples(self.parameter_outfile(rnd), dc)
        return params, data, self.parameter_outfile(rnd)

    def sample(self, checkpoints, **kwargs):
        logging.info(
            f"sampling {self.__class__.__name__} " f" n_rounds={self.config.training.n_rounds}"
        )
        logging.info(f"using observation {self._observed}")
        if not isinstance(checkpoints, list):
            checkpoints = [checkpoints]
        outfiles = []
        for checkpoint in checkpoints:
            params, rnd, obs_fit = self._load(checkpoint)
            self._observed = jnp.asarray(obs_fit).reshape(1, -1)
            assert np.allclose(obs_fit, np.squeeze(self._observed), rtol=1e-03, atol=1e-03)
            snl, model = self.instance()
            rng_key = jr.PRNGKey(self.config.rng_key * self.config.evaluation.rng_key)
            _, outfile = self._sample(rng_key, params, snl, rnd, **kwargs)
            outfiles.append(outfile)
        return outfiles

    def _sample(self, rng_key, params, snl, rnd, **kwargs):
        logging.info(f"sampling with parameter set from iteration {rnd - 1} / round {rnd}")
        samples, diagnostics = snl.sample_posterior(
            rng_key, params, self._observed, **self.config.evaluation.sampler
        )
        dc = {
            "samples": np.asarray(samples),
            "diagnostics": {"ess": diagnostics.ess, "rhat": diagnostics.rhat},
            "observed": np.squeeze(np.asarray(self._observed)),
        }
        self._save_parameters_or_samples(self.particles_outfile(rnd), dc)
        return samples, self.particles_outfile(rnd)

    def get_optimizer(self):
        optimizer = optax.adam(
            learning_rate=self.config.optimizer.params.learning_rate,
            b1=self.config.optimizer.params.b1,
            b2=self.config.optimizer.params.b2,
            eps=self.config.optimizer.params.eps,
        )
        return optimizer

    def _save_parameters_or_samples(self, flname, dc, **kwargs):
        try:
            self._save_pickle(flname, dc)
        except Exception as e:
            logging.error(str(e))
            dc.pop("diagnostics")
            self._save_pickle(flname, dc)
