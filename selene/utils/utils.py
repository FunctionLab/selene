import logging


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
