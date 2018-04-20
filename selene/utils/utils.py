import logging
import sys
import seaborn as sns
import yaml
import numpy as np


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


def get_matrix_from_in_silico_mutagenesis_results(mut_encs, mut_preds, ref_enc, ref_pred=None):
    """
    Turns the results of an in silico saturated mutagenesis into a matrix, where each row is a base
    and each column is a position in the sequence. The value is the prediction value when that position
    in the sequence has been set to that base.
    :param ref_enc: Encoded unmutated sequence.
    :param ref_pred: Prediction for unmutated sequence. Leave as zero to not include these.
    :param mut_encs: Mutant sequences.
    :param mut_preds: Predictions for mutant sequences.
    :return: n*m matrix of predictions by base change.
    """
    if ref_pred is not None:
        mat = ref_enc * ref_pred
    else:
        mat = np.zeros_like(ref_enc)
    for i in range(len(mut_preds)):
        tmp = (mut_encs[i] + ref_enc)
        tmp[tmp > 1] = 1.
        mat += (tmp - ref_enc) * mut_preds[i]
    return mat


def get_plot_from_in_silico_mutagenesis_results(mut_encs, mut_preds, ref_enc, base_arr=None, ref_pred=None):
    """
    Turns the results of an in silico saturated mutagenesis into a matrix, where each row is a base
    and each column is a position in the sequence. The value is the prediction value when that position
    in the sequence has been set to that base. Returns a plots of this as a heatmap.
    :param ref_enc: Encoded unmutated sequence.
    :param ref_pred: Prediction for unmutated sequence. Leave as zero to not include these.
    :param mut_encs: Mutant sequences.
    :param mut_preds: Predictions for mutant sequences.
    :param base_arr: Bases to use as Y labels.
    :return: n*m matrix of predictions by base change.
    """
    mat = get_matrix_from_in_silico_mutagenesis_results(mut_encs, mut_preds, ref_enc, ref_pred)
    if ref_pred is None:
        mask = ref_enc
    else:
        mask = None
    return sns.heatmap(mat, linewidths=0., yticklabels=base_arr, cmap="RdBu_r", mask=mask)


def read_yaml_file(config_file):
    with open(config_file, "r") as config_file:
        try:
            config_dict = yaml.load(config_file)
            return config_dict
        except yaml.YAMLError as exception:
            sys.exit(exception)
