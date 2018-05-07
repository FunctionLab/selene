"""
This class is the abstract base class for handling model predicions
"""
from abc import ABCMeta
from abc import abstractmethod


def write_to_file(data_across_features, info_cols, column_names, filename):
    """Write samples with valid predictions/scores to a tab-delimited file.

    Parameters
    ----------
    data_across_features : list of arraylike
        For each sequence input, we should have predictions or scores derived
        from those predictions across all the genomic/sequence-level features
        our model can predict. The length of this list is the number of
        sequences inputted to the model and the length of each element
        (arraylike) in the list is the number of sequence-level features.
    info_cols : list of arraylike
        Identifying information attached to each sequence entry. Each item
        in `info_cols` corresponds to each row that will be written to the
        file. All columns in an element of `info_cols` will be prepended to
        the values in an element of `data_across_features`.
    column_names : list of str
        The column names written as the first line of the file.
    filename : str
        Filepath to which we should write the predictions/scores.

    Returns
    -------
    None
        Writes information to file
    """
    with open(filename, 'w+') as file_handle:
        file_handle.write("{columns}\n".format(
            columns='\t'.join(column_names)))
        for info, preds in zip(info_cols, data_across_features):
            preds_str = '\t'.join(
                probabilities_to_string(preds))
            info_str = '\t'.join([str(i) for i in info])
            file_handle.write(f"{info_str}\t{preds_str}\n")


def write_NAs_to_file(info_cols, column_names, filename):
    """Writes samples with NA predictions or scores to a tab-delimited file.

    Parameters
    ----------
    info_cols : list of arraylike
        NA entries should still have identifying information attached
        to each entry. Each item in `info_cols` corresponds to each
        row that will be written to the file. The item then, is the list
        of column value that go in this row.
    column_names : list of str
        Column names written as the first line of the file.
    filename : str
        Filepath to which we should write the NAs.

    Returns
    -------
    None
        Writes information to file
    """
    with open(filename, 'w+') as file_handle:
        file_handle.write("{columns}\n".format(
            columns='\t'.join(column_names)))
        for info in info_cols:
            write_info = '\t'.join([str(i) for i in info])
            file_handle.write(f"{write_info}\n")


def probabilities_to_string(probabilities):
    """Converts a list of probability values (floats) to a list of
    str probability values, where each value is represented in
    scientific notation with 2 digits after the decimal.

    Parameters
    ----------
    probabilities : list of float

    Returns
    -------
    list of str
    """
    return ["{:.2e}".format(p) for p in probabilities]


class PredictionsHandler(metaclass=ABCMeta):
    """
    The base class for handling model predictions.
    """

    def __init__(self):
        self.needs_base_pred = False
        self.results = []
        self.samples = []
        self.NA_samples = []

    def handle_NA(self, row_ids):
        """
        Handle rows without data that we still want to write to file.
        """
        self.NA_samples.append(row_ids)

    @abstractmethod
    def handle_batch_predictions(self, *args, **kwargs):
        """
        Must be able to handle a batch of model predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def write_to_file(self, *args, **kwargs):
        """
        Writes accumulated handler results to file.
        """
        raise NotImplementedError
