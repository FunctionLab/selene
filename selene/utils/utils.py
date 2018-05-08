import logging


def load_features_list(features_file):
    """Reads in a file of distinct features line-by-line and returns
    these features as a list

    Parameters
    ----------
    features_file : str
        Path to the features file. Each feature must be on its own line.

    Returns
    -------
    list
        The list of features. Order of features matches that of the
    file (reading from top to bottom).
    """
    features = []
    with open(features_file, 'r') as file_handle:
        for line in file_handle:
            features.append(line.strip())
    return features


def initialize_logger(out_filepath, verbosity=2):
    """This function can only be called successfully once.
    If the logger has already been initialized with handlers,
    the function exits. Otherwise, it proceeds to set the
    logger configurations.
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

    file_handle = logging.FileHandler(out_filepath)
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)
