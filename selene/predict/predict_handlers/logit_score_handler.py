"""
TODO
"""
import numpy as np
from scipy.special import logit

from .handler import write_to_file, PredictionsHandler


class LogitScoreHandler(PredictionsHandler):
    """
    The logit score handler calculates and records the
    difference between `logit(alt)` and `logit(ref)` predictions.
    For reference, if some event occurs with probability :math:`p`,
    then the log-odds is the logit of `p`, or

    .. math::
        \\mathrm{logit}(p) = \\log\\left(\\frac{p}{1 - p}\\right) =
        \\log(p) - \\log(1 - p)

    Attributes
    ----------
    needs_base_pred : bool
        # TODO
    column_names : list(str)
        # TODO
    results : list # TODO
        # TODO
    samples : list # TODO
        # TODO
    NA_samples : list # TODO
        # TODO
    output_path : str
        # TODO

    Parameters
    ----------
    features_list : list of str
        List of sequence-level features, in the same order that the
        model will return its predictions.
    nonfeature_columns : list of str
        Columns in the file that help to identify the input sequence to
        which the features data corresponds.
    output_path : str
        Path to the file where the logit scores will be written.

    Notes
    -----



    """

    def __init__(self,
                 features_list,
                 nonfeature_columns,
                 output_path):
        """
        Constructs a new `LogitScoreHandler` object.
        """
        super(LogitScoreHandler).__init__()

        self.needs_base_pred = True
        self.column_names = nonfeature_columns + features_list
        self.results = []
        self.samples = []
        self.NA_samples = []
        self.output_path = output_path

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
            The baseline prediction(s) used to compute the logit scores.
            This must either be a vector of :math:`N` values, or a
            matrix of shape :math:`B \\times N` (where :math:`B` is
            the size of the mini-batch, and :math:`N` is the number of
            features).

        """
        logits = logit(baseline_predictions) - logit(batch_predictions)
        self.results.append(logits)
        self.samples.append(batch_ids)

    def write_to_file(self):
        """
        TODO
        """
        self.results = np.vstack(self.results)
        self.samples = np.vstack(self.samples)
        write_to_file(self.results,
                      self.samples,
                      self.column_names,
                      self.output_path)
