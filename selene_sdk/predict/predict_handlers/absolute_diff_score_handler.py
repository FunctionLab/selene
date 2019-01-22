"""
Handles computing and outputting the absolute difference scores
"""
import numpy as np

from .handler import _create_warning_handler
from .handler import PredictionsHandler


class AbsDiffScoreHandler(PredictionsHandler):
    """
    The "abs diff score" is the absolute difference between `alt` and `ref`
    predictions.

    Parameters
    ----------
    features : list(str)
        List of sequence-level features, in the same order that the
        model will return its predictions.
    nonfeature_columns : list(str)
        Columns in the file that will help to identify the sequence
        or variant to which the model prediction scores correspond.
    output_path_prefix : str
        Path to the file to which Selene will write the absolute difference
        scores. The path may contain a filename prefix. Selene will append
        `abs_diffs` to the end of the prefix if it exists (otherwise the
        file will be named `abs_diffs.tsv`/`.h5`).
    output_format : {'tsv', 'hdf5'}
        Specify the desired output format. TSV can be specified if the final
        file should be easily perused (e.g. viewed in a text editor/Excel).
        However, saving to a TSV file is much slower than saving to an HDF5
        file.

    Attributes
    ----------
    needs_base_pred : bool
        Whether the handler needs the base (reference) prediction as input
        to compute the final output

    """

    def __init__(self,
                 features,
                 nonfeature_columns,
                 output_path_prefix,
                 output_format):
        """
        Constructs a new `AbsDiffScoreHandler` object.
        """
        super(AbsDiffScoreHandler, self).__init__(
            features, nonfeature_columns, output_path_prefix, output_format)
        self.needs_base_pred = True
        self._results = []
        self._samples = []
        self._NA_samples = []

        self._features = features
        self._nonfeature_columns = nonfeature_columns
        self._output_path_prefix = output_path_prefix
        self._output_format = output_format

        self._create_write_handler("abs_diffs")

        self._warn_handle = None

    def handle_NA(self, batch_ids):
        """
        Handle batch sequence/variant IDs where a full sequence context
        could not be fetched (N/A)

        Parameters
        ----------
        batch_ids : list(str)
            List of sequence/variant identifiers

        """
        super().handle_NA(batch_ids)

    def handle_warning(self,
                       batch_predictions,
                       batch_ids,
                       baseline_predictions):
        """
        TODO

        """
        if self._warn_handle is None:
            self._warn_handle = _create_warning_handler(
                self._features,
                self._nonfeature_columns,
                self._output_path_prefix,
                self._output_format,
                AbsDiffScoreHandler)
        self._warn_handle.handle_batch_predictions(
            batch_predictions, batch_ids, baseline_predictions)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 baseline_predictions):
        """
        # TODO

        Parameters
        ----------
        batch_predictions : arraylike
            The predictions for a batch of sequences. This should have
            dimensions of :math:`B \\times N` (where :math:`B` is the
            size of the mini-batch and :math:`N` is the number of
            features).
        batch_ids : list(arraylike)
            Batch of sequence identifiers. Each element is `arraylike`
            because it may contain more than one column (written to
            file) that together make up a unique identifier for a
            sequence.
        base_predictions : arraylike
            The baseline prediction(s) used to compute the diff scores.
            Must either be a vector of dimension :math:`N` values or a
            matrix of dimensions :math:`B \\times N` (where :math:`B` is
            the size of the mini-batch, and :math:`N` is the number of
            features).

        """
        absolute_diffs = np.abs(baseline_predictions - batch_predictions)
        self._results.append(absolute_diffs)
        self._samples.append(batch_ids)
        if len(self._results) > 100000:
            self.write_to_file()

    def write_to_file(self, close=False):
        """
        TODO

        """
        super().write_to_file(close=close)
