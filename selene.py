"""
Description:
    This script builds the model and trains it using user-specified input data.

Output:
    Saves model to a user-specified output file.

Usage:
    selene.py <config-yml> [--lr=<lr>]
    selene.py -h | --help

Options:
    -h --help               Show this screen.

    <config-yml>            Model-specific parameters
    --lr=<lr>               If training, the optimizer's learning rate
                            [default: None]
"""
import os
import importlib
import sys
from time import strftime

from docopt import docopt
import torch

from selene.utils import load_path
from selene.utils import instantiate
from selene import __version__


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
        from models.non_strand_specific_module import NonStrandSpecific
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


def execute(operations, config, output_dir):
    """
    Execute operations in _Selene_.

    Parameters
    ----------
    operations : list(str)
        The list of operations to carry out in _Selene_.
    config : dict or object
        The loaded configurations from a YAML file.
    output_dir : str
        The path to the directory where all outputs will be saved.

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
                config["model"], train=True, lr=config["lr"])

            sampler_info = configs["sampler"]
            sampler_info.bind(output_dir=output_dir)

            train_model_info = configs["train_model"]

            data_sampler = instantiate(sampler_info)

            train_model_info.bind(
                model=model,
                data_sampler=data_sampler,
                loss_criterion=loss,
                optimizer_class=optim,
                optimizer_kwargs=optim_kwargs,
                output_dir=output_dir)

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
                    data_sampler=data_sampler,
                    output_dir=output_dir)
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
                else:
                    raise ValueError("in silico mutagenesis requires as input "
                                     "the path to the FASTA file ('input_path')"
                                     " or a sequences ('input_sequence'), but "
                                     " found neither.")
            if "prediction" in configs:
                predict_info = configs["prediction"]
                analyze_seqs.get_predictions_for_fasta_file(**predict_info)

if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version=__version__)

    configs = load_path(arguments["<config-yml>"], instantiate=False)
    lr = arguments["--lr"]

    operations = configs.pop("ops")
    output_dir = configs.pop("output_dir")
    os.makedirs(output_dir, exist_ok=True)
    current_run_output_dir = os.path.join(
        output_dir, strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(current_run_output_dir)

    if "lr" not in configs and lr != "None":
        configs["lr"] = float(arguments["--lr"])
    elif "lr" in configs and lr != "None" and "train" in operations:
        print("Warning: learning rate specified in both {0} "
              "and on the command line (--lr={1}). Using the command "
              "line value for training.".format(
                  arguments["<config-yml>"], lr))

    # @TODO: allow users to pass in a random seed, optional.
    # @TODO: Should we force this seed to match the seeds elsewhere?
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    execute(operations, configs, current_run_output_dir)
