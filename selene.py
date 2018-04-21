"""
Description:
    This script builds the model and trains it using user-specified input data.

Output:
    Saves model to a user-specified output file.

Usage:
    selene.py <lr> <config-yml>
    selene.py -h | --help

Options:
    -h --help               Show this screen.

    <lr>                    Choose the optimizer's learning rate
    <config-yml>            Model-specific parameters
"""
import importlib

from docopt import docopt
import torch

from selene.utils import load_path, instantiate

if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")
    lr = float(arguments["<lr>"])

    configs = load_path(arguments["<config-yml>"], instantiate=False)

    ##################################################
    # TRAIN MODEL PARAMETERS
    ##################################################
    model_info = configs["model"]
    sampler_info = configs["sampler"]
    model_controller_info = configs["model_controller"]

    sampler = instantiate(sampler_info)

    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    import_model_from = model_info["import_model_from"]
    model_class_name = model_info["class"]
    use_module = importlib.import_module(import_model_from)
    model_class = getattr(use_module, model_class_name)

    model = model_class(sampler.sequence_length, sampler.n_features)
    print(model)

    if model_info["non_strand_specific"]["use_module"]:
        from models.non_strand_specific_module import NonStrandSpecific
        model = NonStrandSpecific(
            model, mode=model_info["non_strand_specific"]["mode"])

    checkpoint_info = model_controller_info.pop("checkpoint")
    checkpoint_resume = checkpoint_info.pop("resume")
    checkpoint = None
    if checkpoint_resume:
        checkpoint_file = checkpoint_info.pop("model_file")
        print("Resuming training from checkpoint {0}.".format(
            checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    model_controller_info.bind(checkpoint_resume=checkpoint)

    criterion = use_module.criterion()
    optimizer_class, optimizer_args = use_module.get_optimizer(lr)

    batch_size = model_controller_info.keywords["batch_size"] # Would love to find a better way.
    max_steps = model_controller_info.keywords["max_steps"]
    report_stats_every_n_steps = \
        model_controller_info.keywords["report_stats_every_n_steps"]
    n_validation_samples = model_controller_info.keywords["n_validation_samples"]

    model_controller_info.bind(model=model,
                               data_sampler=sampler,
                               loss_criterion=criterion,
                               optimizer_class=optimizer_class,
                               optimizer_args=optimizer_args)
    if "optional_args" in model_controller_info.keywords:
        optional_args = model_controller_info.pop("optional_args")
        model_controller_info.bind(**optional_args)
    runner = instantiate(model_controller_info)

    print("Training model: {0} steps, {1} batch size.".format(
        max_steps, batch_size))
    runner.train_and_validate()
    if configs["evaluate_on_test"]:
        runner.evaluate()
    if configs["save_datasets"]:
        runner.write_datasets_to_file()
