import numpy as np
from absl import logging
from jax import random as jr
from sbi.inference import SNRE_C

from experiments.snpe_experiment import SNPEExperiment


class SNREExperiment(SNPEExperiment):
    def __init__(self, config, data_config, outdir, run_name):
        super().__init__(config, data_config, outdir, run_name)

        data_fn, prior_fns, simulator, _ = data_config.model_fns(data_config.data)
        self._observed = data_fn(jr.PRNGKey(self.data_config.rng_key), self.datafile)
        self._simulator = simulator
        self._prior = prior_fns()

    def get_model(self):
        from ssnl.nn import torch_classifier

        logging.info(f"initializing model {self.config.method_name}")
        model = torch_classifier.make_model(self.config.model)
        return model

    def instance(self):
        model = self.get_model()
        snre = SNRE_C(self._prior, classifier=model, show_progress_bars=False)
        return snre

    def _fit(self, rng_key, rnd, proposal, model: SNRE_C):
        logging.info(f"doing round={rnd}")

        simul_key, rng_key = jr.split(rng_key)
        theta = proposal.sample((self.config.training.n_simulations_per_round,))
        x = self._simulator(simul_key, theta)
        _ = model.append_simulations(theta, x)

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
