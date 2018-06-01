"""
TODO
"""
import numpy as np

from .handler import write_NAs_to_file, write_to_file, PredictionsHandler


class WritePredictionsHandler(PredictionsHandler):
    """
    Collects batches of model predictions and writes all of them
    to file at the end.

    Attributes
    ----------


    Parameters
    ----------
    features_list : list(str)
        List of sequence-level features, in the same order that the
        model will return its predictions.
    nonfeature_columns : list(str)
        Columns in the file that help to identify the input sequence to
        which the features data corresponds.
    output_path : str
        Path to the file where predictions will be written.

    """

    def __init__(self, features_list, nonfeature_columns, output_path):
        """
        Constructs a new `WritePredictionsHandler` object.

        """

        super(WritePredictionsHandler).__init__()

        self.needs_base_pred = False
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
                                 batch_ids):
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
        """
        self.results.append(batch_predictions)
        self.samples.append(batch_ids)

    def write_to_file(self):
        """
        TODO
        """
        if self.NA_samples:
            NA_file_prefix = '.'.join(
                self.output_path.split('.')[:-1])
            write_NAs_to_file(self.NA_samples,
                              self.column_names,
                              "{0}.NA".format(NA_file_prefix))
        self.results = np.vstack(self.results)
        self.samples = np.vstack(self.samples)
        write_to_file(self.results,
                      self.samples,
                      self.column_names,
                      self.output_path)
