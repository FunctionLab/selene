"""
This module provides utility functions that are not tied to specific
classes or concepts, but still perform specific and important roles
across many of the packages modules.

"""
from collections import OrderedDict
import logging
import sys

import numpy as np

from .multi_model_wrapper import MultiModelWrapper


def _is_lua_trained_model(model):
    if hasattr(model, 'from_lua'):
        return model.from_lua
    check_model = model
    if hasattr(model, 'model'):
        check_model = model.model
    elif type(model) == MultiModelWrapper and \
            hasattr(model, 'sub_models'):
        check_model = model.sub_models[0]
    setattr(model, "from_lua", False)
    setattr(check_model, "from_lua", False)
    for m in check_model.modules():
        if "Conv2d" in m.__class__.__name__:
            setattr(model, "from_lua", True)
            setattr(check_model, "from_lua", True)
    return model.from_lua


def get_indices_and_probabilities(interval_lengths, indices):
    """
    Given a list of different interval lengths and the indices of
    interest in that list, weight the probability that we will sample
    one of the indices in `indices` based on the interval lengths in
    that sublist.

    Parameters
    ----------
    interval_lengths : list(int)
        The list of lengths of intervals that we will draw from. This is
        used to weight the indices proportionally to interval length.
    indices : list(int)
        The list of interval length indices to draw from.

    Returns
    -------
    indices, weights : tuple(list(int), list(float)) \
        Tuple of interval indices to sample from and the corresponding
        weights of those intervals.

    """
    select_interval_lens = np.array(interval_lengths)[indices]
    weights = select_interval_lens / float(np.sum(select_interval_lens))

    keep_indices = []
    for index, weight in enumerate(weights):
        if weight > 1e-10:
            keep_indices.append(indices[index])
    if len(keep_indices) == len(indices):
        return indices, weights.tolist()
    else:
        return get_indices_and_probabilities(
            interval_lengths, keep_indices)


def load_model_from_state_dict(state_dict, model):
    """
    Loads model weights that were saved to a file previously by `torch.save`.
    This is a helper function to reconcile state dict keys where a model was
    saved with/without torch.nn.DataParallel and now must be loaded
    without/with torch.nn.DataParallel.

    Parameters
    ----------
    state_dict : collections.OrderedDict
        The state of the model.
    model : torch.nn.Module
        The PyTorch model, a module composed of submodules.

    Returns
    -------
    torch.nn.Module \
        The model with weights loaded from the state dict.

    Raises
    ------
    ValueError
        If model state dict keys do not match the keys in `state_dict`.

    """
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    model_keys = model.state_dict().keys()
    state_dict_keys = state_dict.keys()

    if len(model_keys) != len(state_dict_keys):
        try:
            model.load_state_dict(state_dict, strict=False)
            return model
        except Exception as e:
            raise ValueError("Loaded state dict does not match the model "
                "architecture specified - please check that you are "
                "using the correct architecture file and parameters.\n\n"
                "{0}".format(e))

    new_state_dict = OrderedDict()
    for (k1, k2) in zip(model_keys, state_dict_keys):
        value = state_dict[k2]
        try:
            new_state_dict[k1] = value
        except Exception as e:
            raise ValueError(
                "Failed to load weight from module {0} in model weights "
                "into model architecture module {1}. (If module name has "
                "an additional prefix `model.` it is because the model is "
                "wrapped in `selene_sdk.utils.NonStrandSpecific`. This "
                "error was raised because the underlying module does "
                "not match that expected by the loaded model:\n"
                "{2}".format(k2, k1, e))
    model.load_state_dict(new_state_dict)
    return model


def load_features_list(input_path):
    """
    Reads in a file of distinct feature names line-by-line and returns
    these features as a list. Each feature name in the file must occur
    on a separate line.

    Parameters
    ----------
    input_path : str
        Path to the features file. Each feature in the input file must
        be on its own line.

    Returns
    -------
    list(str) \
        The list of features. The features will appear in the list in
        the same order they appeared in the file (reading from top to
        bottom).

    Examples
    --------
    A file at "input_features.txt", for the feature names :math:`YFP`
    and :math:`YFG` might look like this:
    ::
        YFP
        YFG


    We can load these features from that file as follows:

    >>> load_features_list("input_features.txt")
    ["YFP", "YFG"]

    """
    features = []
    with open(input_path, 'r') as file_handle:
        for line in file_handle:
            features.append(line.strip())
    return features


def initialize_logger(output_path, verbosity=2):
    """
    Initializes the logger for Selene.
    This function can only be called successfully once.
    If the logger has already been initialized with handlers,
    the function exits. Otherwise, it proceeds to set the
    logger configurations.

    Parameters
    ----------
    output_path : str
        The path to the output file where logs will be written.

    verbosity : int, {2, 1, 0}
        Default is 2. The level of logging verbosity to use.

            * 0 - Only warnings will be logged.
            * 1 - Information and warnings will be logged.
            * 2 - Debug messages, information, and warnings will all be\
                  logged.

    """
    logger = logging.getLogger("selene")
    # check if logger has already been initialized
    if len(logger.handlers):
        return

    if verbosity == 0:
        logger.setLevel(logging.WARN)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    elif verbosity == 2:
        logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s")

    file_handle = logging.FileHandler(output_path)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)

    stdout_formatter = logging.Formatter(
        "%(asctime)s - %(message)s")

    stdout_handle = logging.StreamHandler(sys.stdout)
    stdout_handle.setFormatter(stdout_formatter)
    stdout_handle.setLevel(logging.INFO)
    logger.addHandler(stdout_handle)
