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
import importlib.util

from docopt import docopt
import torch

from selene.utils import load_path
from selene.utils import instantiate
from selene import __version__


def initialize_model(model_configs, train=True, lr=None):
    """
    TODO

    Parameters
    ----------
    model_configs : TODO
        TODO
    train : bool
        TODO
    lr : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    import_model_from = model_configs["file"]
    model_class_name = model_configs["class"]

    spec = importlib.util.spec_from_file_location("module", import_model_from)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, model_class_name)

    if train:
        criterion = module.criterion()
        optim_class, optim_kwargs = module.get_optimizer(lr)

    sequence_length = model_configs["sequence_length"]
    n_classes = model_configs["n_classes_to_predict"]
    model = model_class(sequence_length, n_classes)

    if model_configs["non_strand_specific"]["use_module"]:
        from models.non_strand_specific_module import NonStrandSpecific
        model = NonStrandSpecific(
            model, mode=model_configs["non_strand_specific"]["mode"])
    if train:
        return model, criterion, optim_class, optim_kwargs
    else:
        return model


def execute(operation, config, model=None):
    """
    TODO

    Parameters
    ----------
    operation : TODO
        TODO
    config : TODO
        TODO
    model : TODO
        TODO

    Returns
    -------
    TODO
        TODO

    """
    if operation == "train":
        model, loss, optim, optim_kwargs = initialize_model(
            config["model"], train=True, lr=config["lr"])

        sampler_info = configs["sampler"]
        train_model_info = configs["train_model"]

        data_sampler = instantiate(sampler_info)

        train_model_info.bind(
            model=model,
            data_sampler=data_sampler,
            loss_criterion=loss,
            optimizer_class=optim,
            optimizer_kwargs=optim_kwargs)

        trainer = instantiate(train_model_info)
        trainer.train_and_validate()
        if "evaluate_on_test" in config and config["evaluate_on_test"]:
            trainer.evaluate()
        if "save_datasets" in config and config["save_datasets"]:
            trainer.write_datasets_to_file()

    elif operation == "evaluate":
        pass  # @TODO: figure out what this entails.

    elif operation == "analyze":
        if not model:
            model = initialize_model(configs["model"], train=False)

        analyze_seqs_info = configs["analyze_sequences"]
        analyze_seqs_info.bind(model=model)

        analyze_seqs = instantiate(analyze_seqs_info)

        if "variant_effect_prediction" in configs:
            vareff_info = configs["variant_effect_prediction"]
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


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version=__version__)

    configs = load_path(arguments["<config-yml>"], instantiate=False)
    lr = arguments["--lr"]

    operations = configs.pop("ops")

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

    for op in operations:
        execute(op, configs)
