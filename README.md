# Surjective sequential neural likelihood estimation

[![ci](https://github.com/dirmeier/ssnl/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/ssnl/actions/workflows/ci.yaml)
[![arXiv](https://img.shields.io/badge/arXiv-2308.01054-b31b1b.svg)](https://arxiv.org/abs/2308.01054)

## About

This repository contains the Python code for reproducing the experimental results in the manuscript

> Simon Dirmeier, Carlo Albert, Fernando Perez-Cruz, *Simulation-based Inference for High-dimensional Data using Surjective Sequential Neural Likelihood Estimation*, UAI 2025
> [[arXiv]](https://arxiv.org/abs/2308.01054)

- `configs` contains configuration files for the different inferential algorithms,
- `data_and_models` contains implementation of experimental benchmark models,
- `envs` contains requires conda environment files,
- `ssnl` implements the method SSNL and all baselines,
- `experiments` contains source code for SSNL/SNL/SNPE/SNRE with the logic to run the experiments,
- `.*py` files are entry point scripts that execute the experiments.

## Installation

If you run the Snakemake workflow, no installation of dependencies is requires except having access to `conda` on the command line. E.g., check if this works on the command-line interface:

```bash
conda --version
```

If it is showing a conda version, you are set. Otherwise install Miniconda from [here](https://docs.conda.io/projects/miniconda/en/latest/).

## Usage

To run the experiments, make sure you have access to a `zsh` or `bash` shell.
You can either run experiments manually or use Snakemake to run everything in an automated fashion.

### Automatic execution (recommended)

If you want to run all experiments from the manuscript and the appendix you can do it automatically using Snakemake.
*Note*: this will run all experiments which will require a significant amount of compute hours.
First, install Snakemake via:

```bash
pip install snakemake==9.3.0
```

Then, on a HPC cluster use

```bash
snakemake --cluster {sbatch/qsub/bsub} --use-conda --configfile=snake_config.yaml --jobs N_JOBS
```

where `--cluster {sbatch/qsub/bsub}` specifies the command your cluster uses for job management and `--jobs N_JOBS` sets the number of jobs submitted at the same time. For instance, to run on a SLURM cluster:

```bash
snakemake --cluster sbatch --use-conda  --configfile=snake_config.yaml --jobs 100
```

You can also directly specify your job scheduler:

```bash
snakemake \
  --use-conda \
  --slurm \
  --configfile=snake_config.yaml \
  --jobs 100
```

All the rules have default resources set.

### Manual execution (not recommended)

If you want to manually execute all jobs, first install these conda environments.

```bash
conda env create -f envs/jax-environment.yaml -n ssnl-jax
conda env create -f envs/torch-environment.yaml -n ssnl-torch
conda env create -f envs/eval-environment.yaml -n ssnl-eval
```

We demonstrate running a training job for a specific example (SIR).
First, train SSNL using the command below:

```bash
conda activate ssnl-jax
python 01-main.py  \
    --outdir=<<outpath>> \
    --mode=fit \
    --data_config=data_and_models/sir.py \
    --data_config.rng_key=<<seed>> \
    --config=configs/ssnl.py \
    --config.rng_key=<<seed>> \
    --config.model.reduction_factor=0.75 \
    --config.training.n_rounds=15
```

where `<<outpath>>` is a target directory where results will be stored and `<<seed>>` is an int (which needs to be the same between config and data config).
The same call for SNL would look like this:

```bash
conda activate ssnl-jax
python 01-main.py  \
    --outdir=<<outpath>> \
    --mode=fit \
    --data_config=data_and_models/sir.py \
    --data_config.rng_key=<<seed>> \
    --config=configs/snl.py \
    --config.rng_key=<<seed>> \
    --config.training.n_rounds=15
```

For SNRE and SNPE, the call is the same EXCEPT that the data config needs to be suffixed by `_torch`. For instance, for SNPE:

```bash
conda activate ssnl-torch
python 01-main.py  \
    --outdir=<<outpath>> \
    --mode=fit \
    --data_config=data_and_models/sir_torch.py \
    --data_config.rng_key=<<seed>> \
    --config=configs/snpe.py \
    --config.rng_key=<<seed>> \
    --config.training.n_rounds=15
```

The above commands generate several parameter files with suffix `-params.pkl` that contain the trained neural network parameters. To get posterior samples, you would run

```bash
conda activate ssnl-jax
python 01-main.py  \
    --outdir=<<outpath>> \
    --mode=sample \
    --data_config=data_and_models/sir.py \
    --data_config.rng_key=<<seed>> \
    --config=configs/ssnl.py \
    --config.rng_key=<<seed>> \
    --config.model.reduction_factor=0.75 \
    --config.training.n_rounds=<<round>> \
    --checkpoint=<<outpath>>/sbi-sir-ssnl-seed_<<seed>>-reduction_factor=0.75-n_rounds=<<round>>-params.pkl
```

where `<<round>>` specified a round between 1 and 15 (since above` n_rounds` equalled 15). The `checkpoint` is the file that has been created during training. This creates a file  with suffix `posteriors.pkl`.

To draw samples from the posterior using MCMC:

```bash
conda activate ssnl-jax
python 01-main.py  \
    --outdir=<<outpath>> \
    --mode=sample \
    --data_config=data_and_models/sir.py \
    --data_config.rng_key=<<seed>> \
    --config=configs/mcmc.py \
    --config.rng_key=<<seed>>
```

This creates a file  with suffix `posteriors.pkl`, too.

To compute divergences between true and surrogate posterior, call:

```bash
conda activate ssnl-eval
python compute_mmd.py \
  <<outpath>>/sbi-sir-mcmc-seed_<<seed>>>-posteriors.pkl \
  <<outpath>>/sbi-sir-ssnl-seed_<<seed>>-reduction_factor=0.75-n_rounds=<<round>>-posteriors.pkl \
  <<outpath/<<outfile>>
  10000
```

You will need to repeat that for seeds 1-10, the models in `data_and_models` and the different methods (SSNL, SNL, SNASS, SNASSS, SNRE, SNPE).

## Citation

If you find our work relevant to your research, please consider citing:

```
@inproceedings{
  dirmeier2025surjective,
  title={Simulation-based Inference for High-dimensional Data using Surjective Sequential Neural Likelihood Estimation},
  author={Dirmeier, Simon and Albert, Carlo and Perez-Cruz Fernando},
  year={2025},
  booktitle={Proceedings of the Forty-First Conference on Uncertainty in Artificial Intelligence}
}
```

## Author

Simon Dirmeier <a href="mailto:simd @ mailbox org">simd @ mailbox org</a>