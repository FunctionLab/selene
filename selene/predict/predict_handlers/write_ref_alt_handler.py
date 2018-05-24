from .handler import PredictionsHandler
from .write_predictions_handler import WritePredictionsHandler


class WriteRefAltHandler(PredictionsHandler):
    """Used during variant effect prediction. Write the prediction values
    for the reference and alternate sequences in 2 separate files.
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
            Filepath to which the ref and alt predictions are written--
            .ref and .alt are appended to the 2 files that are outputted
            to distinguish the 2.
        """
        super(WriteRefAltHandler).__init__()

        self.needs_base_pred = True
        self.ref_writer = WritePredictionsHandler(
            features_list, nonfeature_columns, "{0}.ref".format(out_filename))
        self.alt_writer = WritePredictionsHandler(
            features_list, nonfeature_columns, "{1}.alt".format(out_filename))

    def handle_NA(self, batch_ids):
        self.ref_writer.handle_NA(batch_ids)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 base_predictions):
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
            The reference predictions. Must be a matrix of dimensions
            [batch_size, n_features].
        """
        self.ref_writer.handle_batch_predictions(
            base_predictions, batch_ids)
        self.alt_writer.handle_batch_predictions(
            batch_predictions, batch_ids)

    def write_to_file(self):
        self.ref_writer.write_to_file()
        self.alt_writer.write_to_file()
