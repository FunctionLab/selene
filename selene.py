"""
Description:
    This script builds the model and trains it using user-specified input data.

Output:
    Saves model to a user-specified output file.

Usage:
    selene.py <import-module> <model-class-name> <lr>
        <paths-yml> <train-model-yml>
        [-s | --stdout] [--verbosity=<level>]
    selene.py -h | --help

Options:
    -h --help               Show this screen.

    <import-module>         Import the module containing the model
    <model-class-name>      Must be a model class in the imported module
    <lr>                    Choose the optimizer's learning rate
    <paths-yml>             Input data and output filepaths
    <train-model-yml>       Model-specific parameters
    -s --stdout             Will also output logging information to stdout
                            [default: False]
    --verbosity=<level>     Logging verbosity level (0=WARN, 1=INFO, 2=DEBUG)
                            [default: 1]
"""
import importlib
import logging
import os
from time import strftime, time

from docopt import docopt
import torch

from selene.model_train import ModelController
from selene.samplers import IntervalsSampler
from selene.utils import initialize_logger, read_yaml_file
from selene.utils import load, load_path, instantiate

if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")
    import_model_from = arguments["<import-module>"]
    model_class_name = arguments["<model-class-name>"]
    use_module = importlib.import_module(import_model_from)
    model_class = getattr(use_module, model_class_name)

    lr = float(arguments["<lr>"])

    paths = read_yaml_file(
        arguments["<paths-yml>"])

    train_model = load_path(arguments["<train-model-yml>"], instantiate=False)


    ##################################################
    # PATHS
    ##################################################
    dir_path = paths["dir_path"]
    files = paths["files"]
    genome_fasta = os.path.join(
        dir_path, files["genome"])
    genomic_features = os.path.join(
        dir_path, files["genomic_features"])
    coords_only = os.path.join(
        dir_path, files["sample_from_regions"])
    distinct_features = os.path.join(
        dir_path, files["distinct_features"])

    output_dir = paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    current_run_output_dir = os.path.join(
        output_dir, strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(current_run_output_dir)

    ##################################################
    # TRAIN MODEL PARAMETERS
    ##################################################
    sampler_info = train_model["sampler"]
    model_controller_info = train_model["model_controller"]

    ##################################################
    # OTHER ARGS
    ##################################################
    to_stdout = arguments["--stdout"]
    verbosity_level = int(arguments["--verbosity"])

    initialize_logger(
        os.path.join(current_run_output_dir, "{0}.log".format(__name__)),
        verbosity=verbosity_level,
        stdout_handler=to_stdout)
    logger = logging.getLogger("selene")

    t_i = time()
    feature_thresholds = None
    if "specific_feature_thresholds" in sampler_info.keywords:
        feature_thresholds = sampler_info.pop("specific_feature_thresholds")
    else:
        feature_thresholds = None
    if "default_threshold" in sampler_info.keywords:
        if feature_thresholds:
            feature_thresholds["default"] = sampler_info.pop("default_threshold")
        else:
            feature_thresholds = sampler_info.pop("default_threshold")

    if feature_thresholds:
        sampler_info.bind(feature_thresholds=feature_thresholds)
    sampler_info.bind(genome=genome_fasta,
                      query_feature_data=genomic_features,
                      distinct_features=distinct_features,
                      intervals_file=coords_only)
    sampler = instantiate(sampler_info)

    t_i_model = time()
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    model = model_class(sampler.sequence_length, sampler.n_features)
    print(model)

    checkpoint_info = model_controller_info.pop("checkpoint")
    checkpoint_resume = checkpoint_info.pop("resume")
    checkpoint = None
    if checkpoint_resume:
        checkpoint_file = checkpoint_info.pop("model_file")
        logger.info("Resuming training from checkpoint {0}.".format(
            checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    model_controller_info.bind(checkpoint_resume=checkpoint)

    criterion = use_module.criterion()
    optimizer_class, optimizer_args = use_module.get_optimizer(lr)

    t_f_model = time()
    logger.debug(
        "Finished initializing the {0} model from module {1}: {2} s".format(
            model.__class__.__name__,
            import_model_from,
            t_f_model - t_i_model))

    logger.info(model)
    logger.info(optimizer_args)


    if feature_thresholds:
        sampler_info.bind(feature_thresholds=feature_thresholds)
    sampler_info.bind(genome=genome_fasta,
                      query_feature_data=genomic_features,
                      distinct_features=distinct_features,
                      intervals_file=coords_only)
    sampler = instantiate(sampler_info)

    batch_size = model_controller_info.keywords["batch_size"] # Would love to find a better way.
    max_steps = model_controller_info.keywords["max_steps"]
    report_metrics_every_n_steps = \
        model_controller_info.keywords["report_metrics_every_n_steps"]
    n_validation_samples = model_controller_info.keywords["n_validation_samples"]

    model_controller_info.bind(model=model,
                               data_sampler=sampler,
                               loss_criterion=criterion,
                               optimizer_class=optimizer_class,
                               optimizer_args=optimizer_args,
                               output_dir=current_run_output_dir)
    if "optional_args" in model_controller_info.keywords:
        optional_args = model_controller_info.pop("optional_args")
        model_controller_info.bind(**optional_args)
    runner = instantiate(model_controller_info)

    logger.info("Training model: {0} steps, {1} batch size.".format(
        max_steps, batch_size))
    runner.train_and_validate()

    t_f = time()
    logger.info("./train_model.py completed in {0} s.".format(t_f - t_i))
