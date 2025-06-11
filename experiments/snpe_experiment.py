import numpy as np
import torch
from absl import logging
from jax import random as jr
from sbi.inference import SNPE

from experiments.experiment import Experiment


class SNPEExperiment(Experiment):
    def __init__(self, config, data_config, outdir, run_name):
        super().__init__(config, data_config, outdir, run_name)

        data_fn, prior_fns, simulator, _ = data_config.model_fns(data_config.data)
        self._observed = data_fn(jr.PRNGKey(self.data_config.rng_key), self.datafile)
        self._simulator = simulator
        self._prior = prior_fns()

    def get_model(self):
        from ssnl.nn import torch_maf

        logging.info(f"initializing model {self.config.method_name}")
        model = torch_maf.make_model(self.config.model)
        return model

    def instance(self):
        model = self.get_model()
        snpe = SNPE(self._prior, density_estimator=model, show_progress_bars=False)
        return snpe

    def fit(self):
        logging.info(
            f"doing {self.__class__.__name__} " f"for n_rounds={self.config.training.n_rounds}"
        )
        rng_key = jr.PRNGKey(self.config.rng_key * self.config.training.rng_key)
        model = self.instance()
        proposal = self._prior
        outfiles = []
        for itr in range(self.config.training.n_rounds):
            proposal, data, outfile = self._fit(jr.fold_in(rng_key, itr), itr + 1, proposal, model)
            outfiles.append(outfile)
        return outfiles

    def _fit(self, rng_key, rnd, proposal, model: SNPE):
        logging.info(f"doing round={rnd}")
        logging.info(f"using observation {self._observed}")

        simul_key, rng_key = jr.split(rng_key)
        theta = proposal.sample((self.config.training.n_simulations_per_round,))
        x = self._simulator(simul_key, theta)
        _ = model.append_simulations(theta, x, proposal=proposal)

        model.train(
            training_batch_size=self.config.training.batch_size,
            learning_rate=self.config.optimizer.params.learning_rate,
            validation_fraction=self.config.training.percentage_data_as_validation_set,
            max_num_epochs=self.config.training.n_iter,
            stop_after_epochs=self.config.training.n_early_stopping_patience,
            retrain_from_scratch=True,
        )

        proposal = model.build_posterior(
            sample_with=self.config.training.sampler.get("sampler", "rejection"),
            mcmc_method="slice",
            mcmc_parameters={
                k: v for k, v in self.config.training.sampler.items() if not k == "sampler"
            },
        ).set_default_x(self._observed)

        summary = model.summary

        dc = {
            "observed": np.squeeze(self._observed.numpy()),
            "round": rnd,
            "params": proposal,
            "losses": np.vstack([summary["training_log_probs"], summary["validation_log_probs"]]).T,
            "new_data": (x.numpy(), theta.numpy()),
        }
        self._save_parameters_or_samples(self.parameter_outfile(rnd), dc)
        return proposal, model, self.parameter_outfile(rnd)

    def sample(self, checkpoints, **kwargs):
        logging.info(
            f"sampling {self.__class__.__name__} " f" n_rounds={self.config.training.n_rounds}"
        )
        logging.info(f"using observation {self._observed}")
        if not isinstance(checkpoints, list):
            checkpoints = [checkpoints]
        outfiles = []
        for checkpoint in checkpoints:
            posterior, rnd, obs_fit = self._load(checkpoint)
            self._observed = torch.asarray(obs_fit, dtype=torch.float32, copy=True)
            assert np.allclose(obs_fit, np.squeeze(self._observed.numpy()))
            rng_key = jr.PRNGKey(self.config.rng_key * self.config.evaluation.rng_key)
            _, outfile = self._sample(rng_key, posterior, rnd, **kwargs)
            outfiles.append(outfile)
        return outfiles

    def _sample(self, rng_key, posterior, rnd, **kwargs):
        logging.info(f"sampling with parameter set from iteration {rnd - 1} / round {rnd}")
        samples = posterior.sample(
            (self.config.evaluation.n_samples,),
            x=self._observed,
        )
        self._save_parameters_or_samples(
            self.particles_outfile(rnd),
            {"samples": samples.numpy(), "observed": np.squeeze(self._observed.numpy())},
        )
        return samples, self.particles_outfile(rnd)

    def _save_parameters_or_samples(self, flname, dc, **kwargs):
        self._save_pickle(flname, dc)
