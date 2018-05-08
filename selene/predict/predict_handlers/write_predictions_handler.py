import numpy as np

from .handler import write_NAs_to_file, write_to_file, PredictionsHandler

class WritePredictionsHandler(PredictionsHandler):
    """Collects batches of model predictions and writes all of them
    to file at the end.
    """

    def __init__(self, features_list, nonfeature_columns, out_filename):
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
            Filepath to which the predictions are written.
        """

        super(WritePredictionsHandler).__init__()

        self.needs_base_pred = False
        self.column_names = nonfeature_columns + features_list
        self.results = []
        self.samples = []
        self.NA_samples = []
        self.out_filename = out_filename

    def handle_NA(self, batch_ids):
        super().handle_NA(batch_ids)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids):
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
        """
        self.results.append(batch_predictions)
        self.samples.append(batch_ids)

    def write_to_file(self):
        if self.NA_samples:
            NA_file_prefix = '.'.join(
                self.out_filename.split('.')[:-1])
            write_NAs_to_file(self.NA_samples,
                              self.column_names,
                              f"{NA_file_prefix}.NA")
        self.results = np.vstack(self.results)
        self.samples = np.vstack(self.samples)
        write_to_file(self.results,
                      self.samples,
                      self.column_names,
                      self.out_filename)
