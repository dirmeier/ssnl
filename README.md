# SSNL

This repository contains code to run the experiments from 
*Simulation-based inference using surjective sequential neural likelihood estimation*.

## Installation

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and create an environment using the
environment file `environment.yaml`. This should install all required dependencies.

## Usage

To run an experiment load the environment and then executre>

```bash
python main.py  \
  --outdir=results/ \
  --mode=fit \
  --config=configs/slcp/surjection.py \
  --config.training.n_rounds=${n_rounds} \
  --config.rng_seq_key=${key}
```

This generates a file that contains trained neural network parameters. To get posterior samples, run

```bash
python main.py  \
  --outdir=results/ \
  --mode=eval \
  --checkpoint=results/slcp-params.pkl \
  --config=configs/slcp/surjection.py \
  --round=${rounds} \
  --config.rng_seq_key=${key}
```

To compute the MMD between the true and the approximate posterior samples execute:


```bash
python compute_mmd.py \
  results/slcp \
  results/slcp/slcp-nuts-exact-posteriors.pkl \
  results/slcp/slcp-df.pkl
```