import pathlib
from datetime import datetime
from typing import List

import jax
import wandb

from absl import app
from absl import flags
from absl import logging
from jax.lib import xla_bridge
from ml_collections import config_flags, config_dict

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "training configuration", lock_config=False
)
flags.DEFINE_string(
    "outdir", None, "out directory, i.e., place where results are written to"
)
flags.DEFINE_enum("mode", 'fit', ["fit", "eval", "both"], "either fit, evaluate or do both")
flags.DEFINE_string("checkpoint", None, "checkpoint to evaluate")
flags.DEFINE_integer("round", -1, "round of sequential algorithm to evaluate")
flags.DEFINE_bool("usewand", False, "registers run to wandb")
flags.mark_flags_as_required(["config", "outdir"])


def init_and_log_jax_env():
    tm = datetime.now().strftime("%Y-%m-%d-%H%M")
    logging.info("file prefix: %s", tm)
    logging.info("----- Checking JAX installation ----")
    logging.info(jax.devices())
    logging.info(jax.default_backend())
    logging.info(xla_bridge.get_backend().platform)
    logging.info("------------------------------------")
    return tm


def log_config(run_name, config):
    def _log(k, v):
        if isinstance(v, config_dict.ConfigDict):
            for kk, vv in v.items():
                _log(k + "--" + kk, vv)
        elif isinstance(v, List):
            for i, vv in enumerate(v):
                _log(k + "--" + str(i), vv)
        else:
            wandb.config[k] = v

    for k, v in config.items():
        _log(k, v)
    wandb.run.name = run_name
    return config


def main(argv):
    del argv
    if FLAGS.usewand:
        run = wandb.init(project="bistom-sbi", config={})
    else:
        run = None
    _ = init_and_log_jax_env()

    outdir = pathlib.Path(FLAGS.outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    if FLAGS.config.model_type in ["bijection", "surjection"]:
        from experiments.surjective_sbi_experiment import SurjectiveSBIExperiment

        experiment_ctor = SurjectiveSBIExperiment
    elif FLAGS.config.model_type == "snpe":
        from experiments.snpe_experiment import SNPEExperiment

        experiment_ctor = SNPEExperiment
    elif FLAGS.config.model_type == "smc":
        from experiments.smc_experiment import SMCExperiment

        experiment_ctor = SMCExperiment
    elif FLAGS.config.model_type == "nuts":
        from experiments.exact_posterior_experiment import ExactPosteriorExperiment

        experiment_ctor = ExactPosteriorExperiment
    else:
        raise ValueError(f"model type '{FLAGS.config.model_type}' not available")

    run_name = f"sbi-{FLAGS.config.experiment_name}-{FLAGS.config.model_type}-seed_{FLAGS.config.rng_seq_key}"
    experiment = experiment_ctor(FLAGS, run, run_name)
    if FLAGS.usewand:
        _ = log_config(experiment.file_suffix(), FLAGS.config)

    if FLAGS.mode == "fit":
        experiment.fit()
    elif FLAGS.mode == "eval":
        experiment.evaluate(FLAGS.checkpoint, round=FLAGS.round)
    elif FLAGS.mode == "both":
        experiment.fit_and_evaluate()


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
