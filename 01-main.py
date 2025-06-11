import pathlib
import pickle
from datetime import datetime

import jax
from absl import app
from absl import flags
from absl import logging
from jax.lib import xla_bridge
from ml_collections import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "training configuration", lock_config=False
)
config_flags.DEFINE_config_file(
    "data_config", None, "data and model configuration", lock_config=False
)
flags.DEFINE_string(
    "outdir", None, "out directory, i.e., place where results are written to"
)
flags.DEFINE_enum(
    "mode", None, ["fit", "sample", "both"], "either fit, sample or do both"
)
flags.DEFINE_boolean(
    'dosequential', True, 'is a sequential evaluation or not'
)
flags.DEFINE_string("checkpoint", None, "checkpoint to sample from")
flags.mark_flags_as_required(["config", "data_config", "outdir", "mode"])


def init_and_log_jax_env():
    tm = datetime.now().strftime("%Y-%m-%d-%H%M")
    logging.info("file prefix: %s", tm)
    logging.info("----- Checking JAX installation ----")
    logging.info(jax.devices())
    logging.info(jax.default_backend())
    logging.info(jax.device_count())
    logging.info(xla_bridge.get_backend().platform)
    logging.info("------------------------------------")
    return tm


def _get_config_from_checkpoint(checkpoint):
    with open(checkpoint, "rb") as handle:
        d = pickle.load(handle)
    return d["config"], d["data_config"]


def main(argv):
    del argv
    _ = init_and_log_jax_env()

    assert (FLAGS.config.rng_key == FLAGS.data_config.rng_key)
    config = FLAGS.config
    data_config = FLAGS.data_config


    if config.method_name in ["snl", "ssnl"]:
        from experiments.snl_experiment import SNLExperiment

        experiment_ctor = SNLExperiment
    elif config.method_name in ["snasss"]:
        from experiments.snasss_experiment import SNASSSExperiment

        experiment_ctor = SNASSSExperiment
    elif config.method_name in ["snass"]:
        from experiments.snass_experiment import SNASSExperiment

        experiment_ctor = SNASSExperiment
    elif config.method_name in ["snpe"]:
        from experiments.snpe_experiment import SNPEExperiment

        experiment_ctor = SNPEExperiment
    elif config.method_name in ["snre"]:
        from experiments.snre_experiment import SNREExperiment

        experiment_ctor = SNREExperiment
    elif config.method_name == "mcmc":
        from experiments.sampling_experiment import SamplingExperiment

        experiment_ctor = SamplingExperiment
    else:
        raise ValueError(f"model type '{FLAGS.config.method_name}' not available")

    if not pathlib.Path(FLAGS.outdir).exists():
        pathlib.Path(FLAGS.outdir).mkdir(parents=True, exist_ok=False)

    run_name = f"sbi-{data_config.experiment_name}-{config.method_name}-seed_{config.rng_key}"
    if not FLAGS.dosequential:
        run_name += f"-n_samples={config.training.n_simulations_per_round}"

    if config.method_name == "ssnl":
        run_name += f"-reduction_factor={config.model.reduction_factor}"

    experiment = experiment_ctor(config, data_config, FLAGS.outdir, run_name)
    if FLAGS.mode == "fit":
        experiment.fit()
    elif FLAGS.mode == "sample":
        experiment.sample(FLAGS.checkpoint)
    elif FLAGS.mode == "both":
        experiment.fit_and_sample()
    else:
        raise ValueError(f"FLAGS.mode {FLAGS.mode} not defined")


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
