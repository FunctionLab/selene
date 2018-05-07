import numpy as np
from scipy.special import logit

from .handler import write_to_file, PredictionsHandler


class LogitScoreHandler(PredictionsHandler):
    """Logit calculates the absolute difference between `logit(alt)` and
    `logit(ref)` predictions.
    """

    def __init__(self,
                 features_list,
                 nonfeature_columns,
                 out_filename):
        """
        Parameters
        ----------
        features_list : list of str
            List of sequence-level features, in the same order that the
            model will return its predictions.
        nonfeature_columns : list of str
            Columns in the file that help to identify the input sequence to
            which the features data corresponds.
        out_filename : str
            Filepath to which the logit scores are written.
        """
        super(LogitScoreHandler).__init__()

        self.needs_base_pred = True
        self.column_names = nonfeature_columns + features_list
        self.results = []
        self.samples = []
        self.NA_samples = []
        self.out_filename = out_filename

    def handle_NA(self, batch_ids):
        super().handle_NA(batch_ids)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 baseline_predictions):
        """
        Parameters
        ----------
        batch_predictions : arraylike
            Dimensions = [batch_size, n_features]. The predictions for a batch
            of sequences.
        batch_ids : list of arraylike
            Batch of sequence identifiers. Each element is arraylike because
            it may contain more than one column (written to file) that
            together make up a unique identifier for a sequence.
        base_predictions : arraylike
            The baseline prediction(s) used to compute the logit scores.
            Must either be a vector of dimension [n_features] or a matrix
            of dimensions [batch_size, n_features].
        """
        absolute_logits = np.abs(
            logit(baseline_predictions) - logit(batch_predictions))
        self.results.append(absolute_logits)
        self.samples.append(batch_ids)

    def write_to_file(self):
        self.results = np.vstack(self.results)
        self.samples = np.vstack(self.samples)
        write_to_file(self.results,
                       self.samples,
                       self.column_names,
                       self.out_filename)
