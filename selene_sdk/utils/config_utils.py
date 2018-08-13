"""
Utilities for loading configurations, instantiating Python objects, and
running operations in _Selene_.

"""
import os
import importlib
import sys
from time import strftime

import torch

from . import instantiate


def initialize_model(model_configs, train=True, lr=None):
    """
    Initialize model (and associated criterion, optimizer)

    Parameters
    ----------
    model_configs : dict
        Model-specific configuration
    train : bool, optional
        Default is True. If `train`, returns the user-specified optimizer
        and optimizer class that can be found within the input model file.
    lr : float or None, optional
        If `train`, a learning rate must be specified. Otherwise, None.


    Returns
    -------
    model, criterion : tuple(torch.nn.Module, torch.nn._Loss) or \
            model, criterion, optim_class, optim_kwargs : \
                tuple(torch.nn.Module, torch.nn._Loss, torch.optim, dict)

        * `torch.nn.Module` - the model architecture
        * `torch.nn._Loss` - the loss function associated with the model
        * `torch.optim` - the optimizer associated with the model
        * `dict` - the optimizer arguments

        The optimizer and its arguments are only returned if `train` is
        True.

    Raises
    ------
    ValueError
        If `train` but the `lr` specified is not a float.

    """
    import_model_from = model_configs["file"]
    model_class_name = model_configs["class"]

    # TODO: would like to find a better way...
    path, filename = os.path.split(import_model_from)
    parent_path, model_dir = os.path.split(path)
    sys.path.append(parent_path)

    module_name = filename.split('.')[0]
    module = importlib.import_module("{0}.{1}".format(model_dir, module_name))
    model_class = getattr(module, model_class_name)

    sequence_length = model_configs["sequence_length"]
    n_classes = model_configs["n_classes_to_predict"]
    model = model_class(sequence_length, n_classes)

    if model_configs["non_strand_specific"]["use_module"]:
        from selene_sdk.utils import NonStrandSpecific
        model = NonStrandSpecific(
            model, mode=model_configs["non_strand_specific"]["mode"])
    criterion = module.criterion()
    if train and isinstance(lr, float):
        optim_class, optim_kwargs = module.get_optimizer(lr)
        return model, criterion, optim_class, optim_kwargs
    elif train:
        raise ValueError("Learning rate must be specified as a float "
                         "but was {0}".format(lr))
    return model, criterion


def execute(operations, configs, output_dir):
    """
    Execute operations in _Selene_.

    Parameters
    ----------
    operations : list(str)
        The list of operations to carry out in _Selene_.
    configs : dict or object
        The loaded configurations from a YAML file.
    output_dir : str or None
        The path to the directory where all outputs will be saved.
        If None, this means that an `output_dir` was not specified
        in the top-level configuration keys. `output_dir` must be
        specified in each class's individual configuration wherever
        it is required.

    Returns
    -------
    None
        Executes the operations listed and outputs any files
        to the dirs specified in each operation's configuration.

    Raises
    ------
    ValueError
        If an expected key in configuration is missing.

    """
    model = None
    trainer = None
    for op in operations:
        if op == "train":
            model, loss, optim, optim_kwargs = initialize_model(
                configs["model"], train=True, lr=configs["lr"])

            sampler_info = configs["sampler"]
            if output_dir is not None:
                sampler_info.bind(output_dir=output_dir)

            train_model_info = configs["train_model"]

            data_sampler = instantiate(sampler_info)

            train_model_info.bind(
                model=model,
                data_sampler=data_sampler,
                loss_criterion=loss,
                optimizer_class=optim,
                optimizer_kwargs=optim_kwargs)
            if output_dir is not None:
                train_model_info.bind(output_dir=output_dir)

            trainer = instantiate(train_model_info)
            trainer.train_and_validate()

        elif op == "evaluate":
            if not model and "evaluate_model" in configs:
                model, loss = initialize_model(
                    configs["model"], train=False)
                sampler_info = configs["sampler"]

                evaluate_model_info = configs["evaluate_model"]

                data_sampler = instantiate(sampler_info)

                evaluate_model_info.bind(
                    model=model,
                    criterion=loss,
                    data_sampler=data_sampler)
                if output_dir is not None:
                    evaluate_model_info.bind(output_dir=output_dir)
                evaluator = instantiate(evaluate_model_info)
                evaluator.evaluate()
            elif trainer is not None:
                trainer.evaluate()

        elif op == "analyze":
            if not model:
                model, _ = initialize_model(
                    configs["model"], train=False)

            analyze_seqs_info = configs["analyze_sequences"]
            analyze_seqs_info.bind(model=model)
            analyze_seqs = instantiate(analyze_seqs_info)

            if "variant_effect_prediction" in configs:
                vareff_info = configs["variant_effect_prediction"]
                if "vcf_files" not in vareff_info:
                    raise ValueError("variant effect prediction requires "
                                     "as input a list of 1 or more *.vcf "
                                     "files ('vcf_files').")
                for filepath in vareff_info.pop("vcf_files"):
                    analyze_seqs.variant_effect_prediction(
                        filepath, **vareff_info)
            if "in_silico_mutagenesis" in configs:
                ism_info = configs["in_silico_mutagenesis"]
                if "input_sequence" in ism_info:
                    analyze_seqs.in_silico_mutagenesis(**ism_info)
                elif "input_path" in ism_info:
                    analyze_seqs.in_silico_mutagenesis_from_file(**ism_info)
                elif "fa_files" in ism_info:
                    for filepath in ism_info.pop("fa_files"):
                        analyze_seqs.in_silico_mutagenesis_from_file(
                            filepath, **ism_info)
                else:
                    raise ValueError("in silico mutagenesis requires as input "
                                     "the path to the FASTA file "
                                     "('input_path') or a sequence "
                                     "('input_sequence') or a list of "
                                     "FASTA files ('fa_files'), but found "
                                     "neither.")
            if "prediction" in configs:
                predict_info = configs["prediction"]
                analyze_seqs.get_predictions_for_fasta_file(**predict_info)


def parse_configs_and_run(configs,
                          create_subdirectory=True,
                          lr=None):
    """
    Method to parse the configuration YAML file and run each operation
    specified.

    Parameters
    ----------
    configs : dict
        The dictionary of nested configuration parameters. Will look
        for the following top-level parameters:

            * `ops`: A list of 1 or more of the values \
            {"train", "evaluate", "analyze"}. The operations specified\
            determine what objects and information we expect to parse\
            in order to run these operations. This is required.
            * `output_dir`: Output directory to use for all the operations.\
            If no `output_dir` is specified, assumes that all constructors\
            that will be initialized (which have their own configurations\
            in `configs`) have their own `output_dir` specified.\
            Optional.
            * `random_seed`: A random seed set for `torch` and `torch.cuda`\
            for reproducibility. Optional.
            * `lr`: The learning rate, if one of the operations in the list is\
            "train".

    create_subdirectory : bool, optional
        Default is True. If `create_subdirectory`, will create a directory
        within `output_dir` with the name formatted as "%Y-%m-%d-%H-%M-%S",
        the date/time this method was run.
    lr : float or None, optional
        Default is None. If "lr" (learning rate) is already specified as a
        top-level key in `configs`, there is no need to set `lr` to a value
        unless you want to override the value in `configs`. Otherwise,
        set `lr` to the desired learning rate if "train" is one of the
        operations to be executed.

    Returns
    -------
    None
        Executes the operations listed and outputs any files
        to the dirs specified in each operation's configuration.

    """
    operations = configs.pop("ops")

    if "train" in operations and "lr" not in configs and lr != "None":
        configs["lr"] = float(lr)
    elif "train" in operations and "lr" in configs and lr != "None":
        print("Warning: learning rate specified in both the "
              "configuration dict and this method's `lr` parameter. "
              "Using the `lr` value input to `parse_configs_and_run` "
              "({0}, not {1}).".format(lr, configs["lr"]))

    current_run_output_dir = None
    if "output_dir" not in configs:
        print("No top-level output directory specified. All constructors "
              "to be initialized (e.g. Sampler, TrainModel) that require "
              "this parameter must have it specified in their individual "
              "parameter configuration.")
    else:
        current_run_output_dir = configs.pop("output_dir")
        os.makedirs(current_run_output_dir, exist_ok=True)
        if "create_subdirectory" in configs:
            create_subdirectory = configs["create_subdirectory"]
        if create_subdirectory:
            current_run_output_dir = os.path.join(
                current_run_output_dir, strftime("%Y-%m-%d-%H-%M-%S"))
            os.makedirs(current_run_output_dir)
        print("Outputs and logs saved to {0}".format(
            current_run_output_dir))

    if "random_seed" in configs:
        seed = configs.pop("random_seed")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        print("Warning: no random seed specified in config file. "
              "Using a random seed ensures results are reproducible.")

    execute(operations, configs, current_run_output_dir)
