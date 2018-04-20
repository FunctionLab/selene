import logging
import sys
import seaborn as sns
import yaml
import numpy as np
import matplotlib.pyplot as plt


def initialize_logger(out_filepath, verbosity=1, stdout_handler=False):
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

    if stdout_handler:
        stream_handle = logging.StreamHandler(sys.stdout)
        stream_handle.setFormatter(formatter)
        logger.addHandler(stream_handle)


def get_in_silico_mutagenesis_heatmap(baseline_seq, baseline_pred, mut_seqs, mut_preds, sequence):
    """
    Turns the results of an in silico saturated mutagenesis into a matrix, where each row is a base
    and each column is a position in the sequence. The value is the prediction value when that position
    in the sequence has been set to that base.
    :param baseline_seq: Baseline string to mutate.
    :param baseline_pred: Prediction for base string.
    :param mut_seqs: Mutant sequences.
    :param mut_preds: Predictions for mutant sequences.
    :param sequence: A genome or some other sequence.
    :return: n*m matrix of predictions by base change.
    """
    baseline_enc = sequence.sequence_to_encoding(baseline_seq)
    ret = baseline_enc * baseline_pred
    for mut_seq, mut_pred in zip(mut_seqs, mut_preds):
        mut_enc = sequence.sequence_to_encoding(mut_seq)
        ret += ((mut_enc - baseline_enc) * mut_pred)
    return sns.heatmap(ret, linewidths=0, yticklabels=sequence.BASES_ARR, cmap="RdBu_r")


def read_yaml_file(config_file):
    with open(config_file, "r") as config_file:
        try:
            config_dict = yaml.load(config_file)
            return config_dict
        except yaml.YAMLError as exception:
            sys.exit(exception)
