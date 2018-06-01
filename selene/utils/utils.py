"""
This module provides utility functions that are not tied to specific
classes or concepts, but still perform specific and important roles
across many of the packages modules.

"""
import logging


def load_features_list(input_path):
    """
    Reads in a file of distinct feature names line-by-line and returns
    these features as a list. Each feature name in the file must occur
    on a separate line.

    Parameters
    ----------
    input_path : str
        Path to the features file. Each feature in the input file must
        on its own line.

    Returns
    -------
    list(str)
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

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handle = logging.FileHandler(output_path)
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)
