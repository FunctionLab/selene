"""
Handles computing and outputting the log fold change scores
"""
from scipy.special import logit

from .handler import _create_warning_handler
from .handler import PredictionsHandler


class LogitScoreHandler(PredictionsHandler):
    """
    The logit score handler calculates and records the
    difference between `logit(alt)` and `logit(ref)` predictions.
    For reference, if some event occurs with probability :math:`p`,
    then the log-odds is the logit of `p`, or

    .. math::
        \\mathrm{logit}(p) = \\log\\left(\\frac{p}{1 - p}\\right) =
        \\log(p) - \\log(1 - p)

    Parameters
    ----------
    features : list of str
        List of sequence-level features, in the same order that the
        model will return its predictions.
    columns_for_ids : list of str
        Columns in the file that help to identify the input sequence to
        which the features data corresponds.
    output_path_prefix : str
        Path to the file to which Selene will write the absolute difference
        scores. The path may contain a filename prefix. Selene will append
        `logits` to the end of the prefix.
    output_format : {'tsv', 'hdf5'}
        Specify the desired output format. TSV can be specified if you
        would like the final file to be easily perused. However, saving
        to a TSV file is much slower than saving to an HDF5 file.
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
        Constructs a new `LogitScoreHandler` object.
        """
        super(LogitScoreHandler, self).__init__(
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

        self._create_write_handler("logits")

        self._warn_handle = None

    def handle_NA(self, batch_ids):
        """
        TODO

        Parameters
        ----------
        batch_ids : # TODO
            # TODO

        """
        super().handle_NA(batch_ids)

    def handle_warning(self,
                       batch_predictions,
                       batch_ids,
                       baseline_predictions):
        if self._warn_handle is None:
            self._warn_handle = _create_warning_handler(
                self._features,
                self._columns_for_ids,
                self._output_path_prefix,
                self._output_format,
                self._write_mem_limit,
                LogitScoreHandler)
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
            The baseline prediction(s) used to compute the logit scores.
            This must either be a vector of :math:`N` values, or a
            matrix of shape :math:`B \\times N` (where :math:`B` is
            the size of the mini-batch, and :math:`N` is the number of
            features).

        """
        logits = logit(baseline_predictions) - logit(batch_predictions)
        self._results.append(logits)
        self._samples.append(batch_ids)
        if self._reached_mem_limit():
            self.write_to_file()

    def write_to_file(self, close=False):
        """
        TODO
        """
        super().write_to_file(close=close)
