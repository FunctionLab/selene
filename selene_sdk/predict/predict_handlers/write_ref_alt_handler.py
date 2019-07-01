"""
Handles writing the ref and alt predictions
"""
import os

from .handler import PredictionsHandler
from .write_predictions_handler import WritePredictionsHandler


class WriteRefAltHandler(PredictionsHandler):
    """
    Used during variant effect prediction. This handler records the
    predicted values for the reference and alternate sequences, and
    stores these values in two separate files.

    Parameters
    ----------
    features : list(str)
        List of sequence-level features, in the same order that the
        model will return its predictions.
    columns_for_ids : list(str)
        Columns in the file that help to identify the input sequence
        to which the features data corresponds.
    output_path_prefix : str
        Path for the file(s) to which Selene will write the ref alt
        predictions. The path may contain a filename prefix. Selene will
        append `ref_predictions` and `alt_predictions` to the end of the
        prefix to distinguish between reference and alternate predictions
        files written.
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
    write_labels : bool, optional
        Default is True. If you initialize multiple write handlers for the
        same set of inputs with output format `hdf5`, set `write_label` to
        False on all handlers except 1 so that only 1 handler writes the
        row labels to an output file.

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
                 write_mem_limit=1500,
                 write_labels=True):
        """
        Constructs a new `WriteRefAltHandler` object.
        """
        super(WriteRefAltHandler, self).__init__(
            features,
            columns_for_ids,
            output_path_prefix,
            output_format,
            output_size=output_size,
            write_mem_limit=write_mem_limit,
            write_labels=write_labels)

        self.needs_base_pred = True
        self._features = features
        self._columns_for_ids = columns_for_ids
        self._output_path_prefix = output_path_prefix
        self._output_format = output_format
        self._write_mem_limit = write_mem_limit
        self._write_labels = write_labels

        output_path, prefix = os.path.split(output_path_prefix)
        ref_filename = "ref"
        alt_filename = "alt"
        if len(prefix) > 0:
            ref_filename = "{0}.{1}".format(prefix, ref_filename)
            alt_filename = "{0}.{1}".format(prefix, alt_filename)
        ref_filepath = os.path.join(output_path, ref_filename)
        alt_filepath = os.path.join(output_path, alt_filename)

        self._ref_writer = WritePredictionsHandler(
            features,
            columns_for_ids,
            ref_filepath,
            output_format,
            output_size=output_size,
            write_mem_limit=write_mem_limit // 2,
            write_labels=write_labels)

        self._alt_writer = WritePredictionsHandler(
            features,
            columns_for_ids,
            alt_filepath,
            output_format,
            output_size=output_size,
            write_mem_limit=write_mem_limit // 2,
            write_labels=False)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 base_predictions):
        """
        Handles the predictions for a batch of sequences.

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
        self._ref_writer.handle_batch_predictions(
            base_predictions, batch_ids)
        self._alt_writer.handle_batch_predictions(
            batch_predictions, batch_ids)

    def write_to_file(self):
        """
        Writes the stored scores to 2 files (1 for ref, 1 for alt).

        """
        self._ref_writer.write_to_file()
        self._alt_writer.write_to_file()
