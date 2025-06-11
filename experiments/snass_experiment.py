import copy

from absl import logging
from jax import numpy as jnp
from jax import random as jr

from experiments.snl_experiment import SNLExperiment
from ssnl.snass.snass import SNASS


class SNASSExperiment(SNLExperiment):
    def get_model(self, leng):
        from ssnl.nn import inference_surjection

        logging.info(f"initializing model {self.config.method_name}")
        data_config = copy.deepcopy(self.data_config.data)
        data_config.ndim_data = leng
        model = inference_surjection.make_model(self.config.model, data_config)
        return model

    def instance(self):
        prior_sampler, prior_fn = self._prior_fns()
        fns = (prior_sampler, prior_fn), self._simulator

        theta_prime = prior_sampler(seed=jr.PRNGKey(0))
        len_theta_prime = len(jnp.squeeze(theta_prime))
        K = 2 * len_theta_prime
        model = self.get_model(K)
        snasss = SNASS(fns, model, K, 50)
        return snasss, model
