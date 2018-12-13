"""
Description:
    The script we use to prepare the directory/files needed to export
    a model to the Kipoi model zoo.

Output:
    Saves model.yaml and TODO to a user-specified output directory.

Usage:
    kipoi_export.py <saved-model-pth> <class-names> <config-yaml> <output-dir>
    kipoi_export.py -h | --help

Options:
    -h --help              Show this screen.

    <saved-model-pth>      Model .pth.tar file output by Selene during
                           training.
    <class-names>          A .txt of class names the model predicts.
    <config-yaml>          The config.yaml file used to fill out
                           `model-template.yaml`.
    <output-dir>           The output directory.
"""
from collections import OrderedDict
import os
import shutil

from docopt import docopt
from jinja2 import Template
import torch
import yaml

def remove_data_parallel_module(state_dict):
    """State dictionary keys that have `module` at the beginning
    are the result of models saved that were trained with
    `torch.nn.DataParallel`. For exporting to kipoi, we remove
    the `module` prefix from state dictionary keys.

    Parameters
    ----------
    state_dict : collections.OrderedDict
        The model's state dict (e.g. `model.state_dict()` or the
        `state_dict` key in the `.pth.tar` file saved by Selene)

    Returns
    -------
    collections.OrderedDict
        The state dict with the keys updated to remove `module`
        from them, if applicable (otherwise the state dict is
        unchanged).
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module" in k:
            k = k[7:]  # remove `module`
        new_state_dict[k] = v
    return new_state_dict


def save_model_state_dict_only(path_to_pth_tar, output_to_pth_tar):
    """Selene saves additional information in addition to the
    model state dictionary in the `.pth.tar` files it outputs.
    The model state dictionary is in a `state_dict` key within
    the dictionary returned by `torch.load`. We retrieve the
    state dict

    Parameters
    ----------
    path_to_pth_tar : str
        Path to the `.pth.tar` file that Selene outputs during
        training.
    output_to_pth_tar : str
        Specify a path to which the output file should be saved,
        e.g. `model_state.pth.tar`.
    """
    state_and_params = torch.load(path_to_pth_tar)
    state_dict = state_and_params["state_dict"]
    state_dict = remove_data_parallel_module(state_dict)
    torch.save(state_dict, output_to_pth_tar)


def export_to_kipoi(config, output_dir):
    template = None
    with open("./model-template.yaml", 'r') as file_handle:
        template = Template(file_handle.read())
    outfile = os.path.join(output_dir, "model.yaml")
    output = template.render(**config)
    with open(outfile, 'w') as file_handle:
        file_handle.write(output)


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")
    saved_model = arguments["<saved-model-pth>"]
    class_names = arguments["<class-names>"]
    config_file = arguments["<config-yaml>"]
    output_dir = arguments["<output-dir>"]

    os.makedirs(output_dir, exist_ok=True)

    config = None
    with open(config_file, 'r') as ymlfile:
        config = yaml.load(ymlfile)
    model_name_file = "{0}.state.pth.tar".format(config["model_name"])
    model_state_pth = os.path.join(
        output_dir, model_name_file)
    save_model_state_dict_only(saved_model, model_state_pth)
    config["model_weights"] = model_name_file

    class_name_file = os.path.basename(class_names)
    class_names_outpath = os.path.join(output_dir, class_name_file)
    shutil.copyfile(class_names, class_names_outpath)
    config["predictor_names"] = class_name_file

    export_to_kipoi(config, output_dir)

