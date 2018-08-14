"""
This class is the abstract base class for all handlers, i.e. objects
that "handle" model predictions.
# TODO: Clarify.
"""
from abc import ABCMeta
from abc import abstractmethod


def write_to_file(data_across_features, info_cols, output_handle, close=False):
    """
    Write samples with valid predictions/scores to a tab-delimited file.

    Parameters
    ----------
    data_across_features : list(arraylike)
        For each sequence input, we should have predictions or scores derived
        from those predictions across all the genomic/sequence-level features
        our model can predict. The length of this list is the number of
        sequences inputted to the model and the length of each element
        (`arraylike`) in the list is the number of sequence-level features.
    info_cols : list(arraylike)
        Identifying information attached to each sequence entry. Each item
        in `info_cols` corresponds to each row that will be written to the
        file. All columns in an element of `info_cols` will be prepended to
        the values in an element of `data_across_features`.
    output_handle : _io.TextIOWrapper
        File handle we use to write the information
    close : bool, optional
        Default is False. Set `close` to True if you are finished writing
        to this file.

    """
    for info, preds in zip(info_cols, data_across_features):
        preds_str = '\t'.join(
            probabilities_to_string(preds))
        info_str = '\t'.join([str(i) for i in info])
        output_handle.write("{0}\t{1}\n".format(info_str, preds_str))
    if close:
        output_handle.close()


def write_NAs_to_file(info_cols, column_names, output_path):
    """
    Writes samples with NA predictions or scores to a tab-delimited file.

    Parameters
    ----------
    info_cols : list(arraylike)
        NA entries should still have identifying information attached
        to each entry. Each item in `info_cols` corresponds to each
        row that will be written to the file. The item then, is the list
        of column value that go in this row.
    column_names : list(str)
        Column names written as the first line of the file.
    output_path : str
        Path to the file where we should write the `NA`'s.

    Returns
    -------
    None
        Writes information to file
    """
    with open(output_path, 'w+') as file_handle:
        file_handle.write("{columns}\n".format(
            columns='\t'.join(column_names)))
        for info in info_cols:
            write_info = '\t'.join([str(i) for i in info])
            file_handle.write("{0}\n".format(write_info))


def probabilities_to_string(probabilities):
    """
    Converts a list of probability values (`float`s) to a list of
    `str` probability values, where each value is represented in
    scientific notation with 2 digits after the decimal.

    Parameters
    ----------
    probabilities : list(float)

    Returns
    -------
    list(str)

    """
    return ["{:.2e}".format(p) for p in probabilities]


class PredictionsHandler(metaclass=ABCMeta):
    """
    The abstract base class for handlers, which "handle" model
    predictions. # TODO(DOCUMENTATION): Elaborate.

    """
    def __init__(self):
        self.needs_base_pred = False
        self._results = []
        self._samples = []
        self._NA_samples = []

    def handle_NA(self, row_ids):
        """
        Handle rows without data that we still want to write to file.

        Parameters
        ----------
        row_ids : # TODO
            # TODO

        """
        self._NA_samples.append(row_ids)

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
