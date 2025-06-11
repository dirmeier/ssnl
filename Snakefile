RESULTS__ = "results"
RESULTS_PATH__ = config[RESULTS__].rstrip("/") + "/"
CONFIG_PATH__ = "configs/"
DATA_CONFIG_PATH__ = "data_and_models/"


seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data = [
    "slcp",

    "beta_glm",
    "sir",
    "lotka_volterra",
    "ornstein_uhlenbeck",

    "lotka_volterra_very_long",
    "ornstein_uhlenbeck_very_long",
]

non_posterior_data = [
    "jansen_rit",
    "solar_dynamo"
]

only_reduce_by_50 = ["gaussian_mixture", "hyperboloid"]
rounds = [1, 2, 3, 4, 5, 10, 15]


rule all:
    input:
        # training models
        expand(
            RESULTS_PATH__
            + "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds={n_rounds}-params.pkl",
            data=data + non_posterior_data + only_reduce_by_50,
            n_rounds=rounds,
            seed=seeds,
            model=["snre", "snpe", "snl", "snass", "snasss"],
        ),
        expand(
            RESULTS_PATH__
            + "{data}/sbi-{data}-{model}-seed_{seed}-reduction_factor={reduc}-n_rounds={n_rounds}-params.pkl",
            data=data + non_posterior_data,
            n_rounds=rounds,
            seed=seeds,
            reduc=[0.25, 0.5, 0.75],
            model=["ssnl"],
        ),
        expand(
            RESULTS_PATH__
            + "{data}/sbi-{data}-{model}-seed_{seed}-reduction_factor={reduc}-n_rounds={n_rounds}-params.pkl",
            data=only_reduce_by_50,
            n_rounds=rounds,
            seed=seeds,
            reduc=[0.5],
            model=["ssnl"],
        ),
        # sampling
        ## mcmc sampling
        expand(
            RESULTS_PATH__ +
            "{data}/sbi-{data}-mcmc-seed_{seed}-posteriors.pkl",
            data=data + only_reduce_by_50,
            seed=seeds,
        ),
        ## sbi sampling
        expand(
            RESULTS_PATH__
            + "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds={n_rounds}-posteriors.pkl",
            data=data + non_posterior_data + only_reduce_by_50,
            n_rounds=rounds,
            seed=seeds,
            model=["snre", "snpe", "snl", "snass", "snasss"],
        ),
        expand(
            RESULTS_PATH__
            + "{data}/sbi-{data}-{model}-seed_{seed}-reduction_factor={reduc}-n_rounds={n_rounds}-posteriors.pkl",
            data=data + non_posterior_data,
            n_rounds=rounds,
            seed=seeds,
            reduc=[0.25, 0.5, 0.75],
            model=["ssnl"],
        ),
        expand(
            RESULTS_PATH__
            + "{data}/sbi-{data}-{model}-seed_{seed}-reduction_factor={reduc}-n_rounds={n_rounds}-posteriors.pkl",
            data=only_reduce_by_50,
            n_rounds=rounds,
            seed=seeds,
            reduc=[0.5],
            model=["ssnl"],
        ),
        # evaluate posteriors
        expand(
            RESULTS_PATH__
            + "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds={n_rounds}-evaluation.pkl",
            data=data + only_reduce_by_50,
            n_rounds=rounds,
            seed=seeds,
            model=["snl", "snpe", "snre", "snass", "snasss"],
        ),
        expand(
            RESULTS_PATH__
            + "{data}/sbi-{data}-{model}-seed_{seed}-reduction_factor={reduc}-n_rounds={n_rounds}-evaluation.pkl",
            data=data,
            n_rounds=rounds,
            seed=seeds,
            reduc=[0.25, 0.5, 0.75],
            model=["ssnl"],
        ),
        expand(
            RESULTS_PATH__
            + "{data}/sbi-{data}-{model}-seed_{seed}-reduction_factor={reduc}-n_rounds={n_rounds}-evaluation.pkl",
            data=only_reduce_by_50,
            n_rounds=rounds,
            seed=seeds,
            reduc=[0.5],
            model=["ssnl"],
        ),


ruleorder: fit_ssnl_models > fit_snl_models > fit_models
ruleorder: sample_mcmc > sample_ssnl_models > sample_models
ruleorder: eval_posteriors_ssnl > eval_posteriors


################################################################################
# Fit models
################################################################################

rule fit_snl_models:
    input:
        config = CONFIG_PATH__ + "snl.py",
        data_config = DATA_CONFIG_PATH__ + "{data}.py",
    output:
        RESULTS_PATH__ + "{data}/sbi-{data}-data-seed_{seed}.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-n_rounds=1-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-n_rounds=2-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-n_rounds=3-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-n_rounds=4-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-n_rounds=5-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-n_rounds=10-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-n_rounds=15-params.pkl",
    resources:
        tasks = 1,
        cpus_per_task = 10,
        mem_mb = 20000,
        runtime = 60 * 24,
    conda:
        "envs/jax-environment.yaml"
    shell:
        "\
        python 01-main.py  \
         --outdir={RESULTS_PATH__}{wildcards.data}/ \
         --mode=fit \
         --data_config={input.data_config} \
         --data_config.rng_key={wildcards.seed} \
         --config={input.config} \
         --config.rng_key={wildcards.seed} \
         --config.training.n_rounds=15"


rule fit_ssnl_models:
    input:
        config = CONFIG_PATH__ + "ssnl.py",
        data_config = DATA_CONFIG_PATH__ + "{data}.py",
    output:
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-reduction_factor={reduc}-n_rounds=1-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-reduction_factor={reduc}-n_rounds=2-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-reduction_factor={reduc}-n_rounds=3-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-reduction_factor={reduc}-n_rounds=4-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-reduction_factor={reduc}-n_rounds=5-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-reduction_factor={reduc}-n_rounds=10-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-ssnl-seed_{seed}-reduction_factor={reduc}-n_rounds=15-params.pkl",
    resources:
        tasks = 1,
        cpus_per_task = 10,
        mem_mb = 20000,
        runtime = 60 * 24,
    conda:
        "envs/jax-environment.yaml"
    shell:
        "\
        python 01-main.py  \
         --outdir={RESULTS_PATH__}{wildcards.data}/ \
         --mode=fit \
         --data_config={input.data_config} \
         --data_config.rng_key={wildcards.seed} \
         --config={input.config} \
         --config.rng_key={wildcards.seed} \
         --config.model.reduction_factor={wildcards.reduc} \
         --config.training.n_rounds=15"


rule fit_models:
    input:
        RESULTS_PATH__ + "{data}/sbi-{data}-data-seed_{seed}.pkl",
        config = CONFIG_PATH__ + "{model}.py",
        data_config = DATA_CONFIG_PATH__ + "{data}.py"
    output:
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds=1-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds=2-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds=3-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds=4-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds=5-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds=10-params.pkl",
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds=15-params.pkl",
    resources:
        tasks = 1,
        cpus_per_task = 10,
        mem_mb = 20000,
        runtime = 60 * 24,
    conda:
        "envs/jax-environment.yaml"
    shell:
        "\
        python 01-main.py  \
         --outdir={RESULTS_PATH__}{wildcards.data}/ \
         --mode=fit \
         --data_config={input.data_config} \
         --data_config.rng_key={wildcards.seed} \
         --config={input.config} \
         --config.rng_key={wildcards.seed} \
         --config.training.n_rounds=15"

################################################################################
# Sample models
################################################################################

rule sample_ssnl_models:
    input:
        config = CONFIG_PATH__ + "ssnl.py",
        checkpoint = RESULTS_PATH__ + "{data}/sbi-{data}-ssnl-seed_{seed}-reduction_factor={reduc}-n_rounds={n_rounds}-params.pkl",
        data_config = DATA_CONFIG_PATH__ + "{data}.py",
    output:
        RESULTS_PATH__ + "{data}/sbi-{data}-ssnl-seed_{seed}-reduction_factor={reduc}-n_rounds={n_rounds}-posteriors.pkl",
    resources:
        tasks = 1,
        cpus_per_task = 10,
        mem_mb = 20000,
        runtime = 4 * 60,
    conda:
        "envs/jax-environment.yaml"
    shell:
        "\
        python 01-main.py  \
         --outdir={RESULTS_PATH__}{wildcards.data}/ \
         --mode=sample \
         --data_config={input.data_config} \
         --data_config.rng_key={wildcards.seed} \
         --config={input.config} \
         --config.rng_key={wildcards.seed} \
         --config.model.reduction_factor={wildcards.reduc} \
         --config.training.n_rounds={wildcards.n_rounds} \
         --checkpoint={input.checkpoint}"


rule sample_models:
    input:
        config = CONFIG_PATH__ + "{model}.py",
        checkpoint = RESULTS_PATH__ + \
            "{data}/sbi-{data}-snl-seed_{seed}-n_rounds={n_rounds}-params.pkl",
        data_config = DATA_CONFIG_PATH__ + "{data}.py",
    output:
        RESULTS_PATH__ + "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds={n_rounds}-posteriors.pkl",
    resources:
        tasks = 1,
        cpus_per_task = 10,
        mem_mb = 20000,
        runtime = 4 * 60,
    conda:
        "envs/jax-environment.yaml"
    shell:
        "\
        python 01-main.py  \
         --outdir={RESULTS_PATH__}{wildcards.data}/ \
         --mode=sample \
         --data_config={input.data_config} \
         --data_config.rng_key={wildcards.seed} \
         --config={input.config} \
         --config.rng_key={wildcards.seed} \
         --config.training.n_rounds={wildcards.n_rounds} \
         --checkpoint={input.checkpoint}"


rule sample_mcmc:
    input:
        config = CONFIG_PATH__ + "mcmc.py",
        data_config = DATA_CONFIG_PATH__ + "{data}.py",
    output:
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-mcmc-seed_{seed, \d+}-posteriors.pkl",
    resources:
        tasks = 1,
        cpus_per_task = 10,
        mem_mb = 20000,
        runtime = 4 * 60,
    conda:
        "envs/jax-environment.yaml"
    shell:
        "\
        python 01-main.py  \
         --outdir={RESULTS_PATH__}/{wildcards.data}/ \
         --mode=sample \
         --data_config={input.data_config} \
         --data_config.rng_key={wildcards.seed} \
         --config={input.config} \
         --config.rng_key={wildcards.seed}"


################################################################################
# Evaluate posteriors
################################################################################

rule eval_posteriors_ssnl:
    input:
        exact_posterior = RESULTS_PATH__ + \
            "{data}/sbi-{data}-mcmc-seed_{seed}-posteriors.pkl",
        approximate_posterior = RESULTS_PATH__
        + "{data}/sbi-{data}-ssnl-seed_{seed}-reduction_factor={reduc}-n_rounds={n_rounds}-posteriors.pkl",
    output:
        RESULTS_PATH__
        + "{data}/sbi-{data}-ssnl-seed_{seed}-reduction_factor={reduc}-n_rounds={n_rounds}-evaluation.pkl",
    resources:
        tasks = 1,
        cpus_per_task = 10,
        mem_mb = 20000,
        runtime = 6 * 60,
    conda:
        "envs/eval-environment.yaml"
    shell:
        "\
        python 02-evaluate_posteriors.py  \
         {input.approximate_posterior} \
         {input.exact_posterior} \
         {output} \
         10000"


rule eval_posteriors:
    input:
        exact_posterior = RESULTS_PATH__ + \
            "{data}/sbi-{data}-mcmc-seed_{seed}-posteriors.pkl",
        approximate_posterior = RESULTS_PATH__
        + "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds={n_rounds}-posteriors.pkl",
    output:
        RESULTS_PATH__ + \
            "{data}/sbi-{data}-{model}-seed_{seed}-n_rounds={n_rounds}-evaluation.pkl",
    resources:
        tasks = 1,
        cpus_per_task = 10,
        mem_mb = 20000,
        runtime = 6 * 60,
    conda:
        "envs/eval-environment.yaml"
    shell:
        "\
        python 02-evaluate_posteriors.py  \
         {input.approximate_posterior} \
         {input.exact_posterior} \
         {output} \
         10000"
