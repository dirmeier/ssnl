import abc
import os
import pickle

from absl import logging


class Experiment(metaclass=abc.ABCMeta):
    def __init__(self, config, data_config, outdir, run_name):
        logging.info(
            f"running experiment {data_config.experiment_name} with model {config.method_name}"
        )
        self.config = config
        self.data_config = data_config
        self.outdir = outdir
        self.run_name = run_name

    def fit_and_sample(self):
        outfiles = self.fit()
        logging.info("done fitting")
        self.sample(outfiles)
        logging.info("done evaluating")

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def sample(self, checkpoint, **kwargs):
        pass

    def _load(self, outname):
        with open(outname, "rb") as handle:
            d = pickle.load(handle)
        return d["params"], d["round"], d["observed"]

    def particles_outfile(self, rnd):
        outname = self.run_name + f"-n_rounds={rnd}-posteriors.pkl"
        outname = os.path.join(self.outdir, outname)
        return outname

    def parameter_outfile(self, rnd):
        outname = self.run_name + f"-n_rounds={rnd}-params.pkl"
        outname = os.path.join(self.outdir, outname)
        return outname

    def _save_pickle(self, filename, obj):
        with open(filename, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def datafile(self):
        return os.path.join(
            self.outdir,
            f"sbi-{self.data_config.experiment_name}-data-seed_{self.config.rng_key}.pkl",
        )
