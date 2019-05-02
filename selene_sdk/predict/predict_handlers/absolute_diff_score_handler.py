"""
Handles computing and outputting the absolute difference scores
"""
import numpy as np

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
    columns_for_ids : list(str)
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
    output_size : int, optional
        The total number of rows in the output. Must be specified when
        the output_format is hdf5.
    write_mem_limit : int, optional
        Default is 1500. Specify the amount of memory you can allocate to
        storing model predictions/scores for this particular handler, in MB.
        Handler will write to file whenever this memory limit is reached.

    Attributes
    ----------
    needs_base_pred : bool
        Whether the handler needs the base (reference) prediction as input
        to compute the final output

    """

    def __init__(self,
                 features,
                 columns_for_ids,
                 output_path_prefix,
                 output_format,
                 output_size=None,
                 write_mem_limit=1500):
        """
        Constructs a new `AbsDiffScoreHandler` object.
        """
        super(AbsDiffScoreHandler, self).__init__(
            features,
            columns_for_ids,
            output_path_prefix,
            output_format,
            output_size,
            write_mem_limit)
        self.needs_base_pred = True
        self._results = []
        self._samples = []
        self._NA_samples = []

        self._features = features
        self._columns_for_ids = columns_for_ids
        self._output_path_prefix = output_path_prefix
        self._output_format = output_format
        self._write_mem_limit = write_mem_limit

        self._create_write_handler("abs_diffs")

    def handle_NA(self, batch_ids):
        """
        Handles batch sequence/variant IDs where a full sequence context
        could not be fetched (N/A)

        Parameters
        ----------
        batch_ids : list(str)
            List of sequence/variant identifiers

        """
        super().handle_NA(batch_ids)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 baseline_predictions):
        """
        Handles the model predictions for a batch of sequences. Computes the
        absolute difference between the predictions for 1 or a batch of
        reference sequences and a batch of alternate sequences (i.e. sequences
        that are slightly changed/mutated from the reference).

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
        if self._reached_mem_limit():
            self.write_to_file()

    def write_to_file(self):
        """
        Writes stored scores to a file.

        """
        super().write_to_file()
