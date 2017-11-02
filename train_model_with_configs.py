"""
Description:
    This script builds the model and trains it using user-specified input data.

Output:
    Saves model to a user-specified output file.

Usage:
    train_model_with_configs.py <import-model> <optimizer> <lr>
        <paths-yml> <train-model-yml>
        [--runs=<n-runs>]
        [-s | --stdout] [-v | --verbose]
    train_model_with_configs.py -h | --help

Options:
    -h --help               Show this screen.

    <import-model>          Choose a model module to import. Must be able
                            to access a model with the class name
                            "SeqModel" from this module.
    <optimizer>             Choose an optimizer (SGD, Adam, RMSprop)
    <lr>                    Choose the optimizer's learning rate
    <paths-yml>             Input data and output filepaths
    <train-model-yml>       DeepSEA model-specific parameters

    --runs=<n-runs>         Specify number of times to do a full run of the
                            model training. (Will initialize the model using
                            a different random seed, from 0 to <n-runs>, each
                            time
                            [default: 1]
    -s --stdout             Will also output logging information to stdout
                            [default: False]
    -v --verbose            Include debug messages in logging information
                            [default: False]
"""
import importlib
import logging
import os
import sys
from time import strftime, time

from docopt import docopt
import torch
from torch import nn

from model_controller import ModelController
from sampler import ChromatinFeaturesSampler
from utils import read_yaml_file

if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")

    import_model_from = arguments["<import-model>"]
    model = importlib.import_module(import_model_from)

    optimizer = arguments["<optimizer>"]
    lr = float(arguments["<lr>"])

    paths = read_yaml_file(
        arguments["<paths-yml>"])
    train_model = read_yaml_file(
        arguments["<train-model-yml>"])

    ##################################################
    # PATHS
    ##################################################
    genome_fa_file = paths["genome"]

    features_paths = paths["features"]
    features_dir = features_paths["dir_path"]
    features_files = features_paths["filenames"]
    genomic_features = os.path.join(
        features_dir, features_files["genomic_features"])
    coords_only = os.path.join(
        features_dir, features_files["coords_only"])
    distinct_features = os.path.join(
        features_dir, features_files["distinct_features"])

    output_dir = paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    current_run_output_dir = os.path.join(
        output_dir, strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(current_run_output_dir)

    ##################################################
    # TRAIN MODEL PARAMETERS
    ##################################################
    #optimizer_info = train_model["optimizer"]
    sampler_info = train_model["sampler"]
    model_controller_info = train_model["model_controller"]

    ##################################################
    # OTHER ARGS
    ##################################################
    n_runs = int(arguments["--runs"])
    to_stdout = arguments["--stdout"]
    verbose = arguments["--verbose"]

    log = logging.getLogger("deepsea")
    if verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # TODO: LOOP FOR ALL THE RUNS.
    # ALSO this wouldn't need to happen linearly.
    # Should start diff jobs on diff GPU nodes for each of the runs.

    log_output = os.path.join(current_run_output_dir, "train_model_log.out")
    file_handle = logging.FileHandler(log_output)
    file_handle.setFormatter(formatter)
    log.addHandler(file_handle)

    if to_stdout:
        stream_handle = logging.StreamHandler(sys.stdout)
        stream_handle.setFormatter(formatter)
        log.addHandler(stream_handle)

    t_i = time()
    print(sampler_info["optional_args"])
    sampler = ChromatinFeaturesSampler(
        genome_fa_file,
        genomic_features,
        coords_only,
        distinct_features,
        sampler_info["holdout_test"],
        sampler_info["holdout_validate"],
        #current_run_output_dir,
        **sampler_info["optional_args"])

    t_i_model = time()

    model = model.DeepSEA(sampler.window_size, sampler.n_features)

    checkpoint_info = model_controller_info["checkpoint"]
    checkpoint_resume = checkpoint_info["resume"]
    checkpoint = None
    if checkpoint_resume:
        checkpoint_file = checkpoint_info["model_file"]
        log.info("Resuming training from checkpoint {0}.".format(
            checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

    # TODO: should this be specified elsewhere? criterion is dependent
    # on the output of your model
    # though it may be argued that most sequence-level deep learning models
    # (at least those that use this tool) will follow this format...
    criterion = nn.BCELoss()
    optimizer_args = {
        "use_optim": optimizer,
        "lr": lr
    }

    #optimizer_args = {
    #    "use_optim": optimizer_info["name"],
    #    "lr": float(optimizer_info["learning_rate"])
    #}
    t_f_model = time()
    log.debug(
        "Finished initializing the {0} model from module {1}: {2} s".format(
            model.__class__.__name__,
            import_model_from,
            t_f_model - t_i_model))

    log.info(model)
    log.info(optimizer_args)

    batch_size = model_controller_info["batch_size"]
    n_epochs = model_controller_info["n_epochs"]
    n_total_validate = model_controller_info["n_total_validate"]
    n_train_batch_per_epoch = model_controller_info["n_train_batch_per_epoch"]

    runner = ModelController(
        model, sampler, criterion, optimizer_args,
        batch_size, n_total_validate,
        n_train_batch_per_epoch,
        current_run_output_dir,
        checkpoint_resume=checkpoint,
        **model_controller_info["optional_args"])

    #stdout_logger_fields = []
    #runner.register_plugin(Logger(stdout_logger_fields,
    #                              interval=[(20, "iteration"), (1, "epoch")]))
    log.info("Training model: {0} epochs, {1} batch size.".format(
        n_epochs, batch_size))
    runner.train_and_validate(n_epochs)

    t_f = time()
    log.info("./train_model.py completed in {0} s.".format(t_f - t_i))