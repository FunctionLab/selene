"""
TODO
"""
import numpy as np

from .handler import write_to_file, PredictionsHandler


class AbsDiffScoreHandler(PredictionsHandler):
    """
    The "abs diff score" is the absolute difference between `alt` and `ref`
    predictions.

    Attributes
    ----------
    needs_base_pred : bool
        # TODO
    column_names : list(str)
            # TODO
    results : list        # TODO
            # TODO
    samples : list        # TODO
            # TODO
    NA_samples : list        # TODO
            # TODO
    output_path : str
            # TODO

    Parameters
    ----------
    features : list(str)
        List of sequence-level features, in the same order that the
        model will return its predictions.
    nonfeature_columns : list(str)
        Columns in the file that help to identify the input sequence to
        which the features data corresponds.
    output_path : str
        Path to the file to write diff scores to.
    """

    def __init__(self,
                 features,
                 nonfeature_columns,
                 output_path):
        """
        Constructs a new `AbsDiffScoreHandler` object.
        """
        super(AbsDiffScoreHandler, self).__init__()

        self.needs_base_pred = True
        self._results = []
        self._samples = []
        self._NA_samples = []
        column_names = nonfeature_columns + features
        self._output_handle = open(output_path, 'w+')
        self._output_handle.write("{0}\n".format(
            '\t'.join(column_names)))

    def handle_NA(self, batch_ids):
        """
        TODO

        Parameters
        ----------
        batch_ids : # TODO
            # TODO

        """
        super().handle_NA(batch_ids)

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
        if len(self._results) > 200000:
            self.write_to_file()

    def write_to_file(self, close=False):
        """
        TODO

        """
        if not self._results:
            self._output_handle.close()
            return None
        self._results = np.vstack(self._results)
        self._samples = np.vstack(self._samples)
        write_to_file(self._results,
                      self._samples,
                      self._output_handle,
                      close=close)
        self._results = []
        self._samples = []
