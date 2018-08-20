"""
TODO
"""
from .handler import PredictionsHandler
from .write_predictions_handler import WritePredictionsHandler


class WriteRefAltHandler(PredictionsHandler):
    """
    Used during variant effect prediction. This handler records the
    predicted values for the reference and alternate sequences, and
    stores these values in two separate files.

    Attributes
    ----------
    ref_writer : # TODO
        # TODO
    alt_writer : # TODO
        # TODO

    Parameters
    ----------
    features : list(str)
        List of sequence-level features, in the same order that the
        model will return its predictions.
    nonfeature_columns : list(str)
        Columns in the file that help to identify the input sequence
        to which the features data corresponds.
    out_filename : str
        Path to the file where the reference and alternate sequences'
        predictions are written. To distinguish between these two
        files, `.ref` and `.alt` will be used as the suffixes for
        the file names.
    """

    def __init__(self, features, nonfeature_columns, out_filename):
        """
        Constructs a new `WriteRefAltHandler` object.
        """
        super(WriteRefAltHandler).__init__()

        self.needs_base_pred = True
        self.ref_writer = WritePredictionsHandler(
            features, nonfeature_columns, "{0}.ref".format(out_filename))
        self.alt_writer = WritePredictionsHandler(
            features, nonfeature_columns, "{0}.alt".format(out_filename))

    def handle_NA(self, batch_ids):
        """
        TODO

        Parameters
        ----------
        batch_ids : TODO
            TODO

        """
        self.ref_writer.handle_NA(batch_ids)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 base_predictions):
        """
        TODO

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
        self.ref_writer.handle_batch_predictions(
            base_predictions, batch_ids)
        self.alt_writer.handle_batch_predictions(
            batch_predictions, batch_ids)

    def write_to_file(self, close=False):
        """
        TODO
        """
        self.ref_writer.write_to_file(close=close)
        self.alt_writer.write_to_file(close=close)
