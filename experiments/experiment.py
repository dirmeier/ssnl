import abc
import os
import pickle

from absl import logging


class Experiment(metaclass=abc.ABCMeta):
    def __init__(self, FLAGS, run, run_name):
        config = FLAGS.config
        logging.info(
            f"running experiment {config.experiment_name} with model {config.model_type}"
        )
        self.FLAGS = FLAGS
        self.config = config
        self.run = run
        self.run_name = run_name

    def fit_and_evaluate(self):
        self.fit()
        logging.info("done fitting")
        self.evaluate(self.training_outfile())
        logging.info("done evaluating")

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def evaluate(self, checkpoint, **kwargs):
        pass

    def save(self, **kwargs):
        pass

    def _load(self, outname):
        with open(outname, "rb") as handle:
            d = pickle.load(handle)
        self.config = d["config"]
        return d["params"]

    def training_outfile(self):
        return self.file_name() + "-params.pkl"

    def validation_outfile(self):
        return self.file_name() + "-validation.pkl"

    def posteriors_outfile(self):
        return self.file_name() + "-posteriors.pkl"

    def file_name(self):
        return os.path.join(self.FLAGS.outdir, self.file_suffix())
