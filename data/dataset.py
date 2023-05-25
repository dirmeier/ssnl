from absl import logging


def load_dataset(rng_seq_key, config):
    logging.info(f"loading data set: {config.dataset}")

    if "slcp" in config.dataset and config.model_type != "snpe":
        from data import slcp

        model = slcp.slcp_model
    elif "slcp" in config.dataset and config.model_type == "snpe":
        from data import slcp_torch

        model = slcp_torch.slcp_model
    elif config.experiment_name == "solar_dynamo" and config.model_type != "snpe":
        from data import solar_dynamo

        model = solar_dynamo.solar_dynamo_model
    elif config.experiment_name == "solar_dynamo" and config.model_type == "snpe":
        from data import solar_dynamo_torch

        model = solar_dynamo_torch.solar_dynamo_model
    elif config.experiment_name == "sun_spots" and config.model_type != "snpe":
        from data import sun_spots

        model = sun_spots.sunspot_model
    elif config.experiment_name == "sun_spots" and config.model_type == "snpe":
        from data import sun_spots_torch

        model = sun_spots_torch.sunspot_model

    elif config.experiment_name == "factor_model" and config.model_type != "snpe":
        from data import factor_model

        model = factor_model.factor_model
    elif config.experiment_name == "factor_model" and config.model_type == "snpe":
        from data import factor_model_torch

        model = factor_model_torch.factor_model

    elif config.experiment_name == "non_linear_factor_model" and config.model_type != "snpe":
        from data import non_linear_factor_model

        model = non_linear_factor_model.non_linear_factor_model
    elif config.experiment_name == "non_linear_factor_model" and config.model_type == "snpe":
        from data import non_linear_factor_model_torch

        model = non_linear_factor_model_torch.non_linear_factor_model

    else:
        logging.fatal(f"didn't find dataset: {config.dataset}")
        raise ValueError(f"didn't find dataset: {config.dataset}")

    return model(config.data)
