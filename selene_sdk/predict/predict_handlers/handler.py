"""
This class is the abstract base class for all handlers, i.e. objects
that "handle" model predictions. Specifically, handlers should store
the model predictions or scores derived from those predictions and eventually
output them according to a user-specified output format.
"""
from abc import ABCMeta
from abc import abstractmethod
import os
from sys import getsizeof

import h5py

def write_to_tsv_file(data_across_features, info_cols, output_filepath):
    """
    Write samples with valid predictions/scores to a tab-delimited file.

    Parameters
    ----------
    data_across_features : list(arraylike)
        For each sequence input, we should have predictions or scores derived
        from those predictions across all the genomic/sequence-level features
        our model can predict. The length of this list is the number of
        sequences inputted to the model and the length of each element
        (`arraylike`) in the list is the number of sequence-level features.
    info_cols : list(arraylike)
        Identifying information attached to each sequence entry. Each item
        in `info_cols` corresponds to each row that will be written to the
        file. All columns in an element of `info_cols` will be prepended to
        the values in an element of `data_across_features`.
    output_handle : _io.TextIOWrapper
        File handle we use to write the information

    """
    with open(output_filepath, 'a') as output_handle:
        for info_batch, preds_batch in zip(info_cols, data_across_features):
            for info, preds in zip(info_batch, preds_batch):
                preds_str = '\t'.join(
                    probabilities_to_string(list(preds)))
                info_str = '\t'.join([str(i) for i in info])
                output_handle.write("{0}\t{1}\n".format(info_str, preds_str))


def write_to_hdf5_file(data_across_features,
                       info_cols,
                       hdf5_filepath,
                       info_filepath,
                       start_index):
    """
    Write samples with valid predictions/scores to an HDF5 file. The
    dataset attached to this file will be accessed using the key "data".
    Each column corresponds to the prediction/score for a model class
    (e.g. genomic feature), and each row is a different input
    variant/sequence.

    Parameters
    ----------
    data_across_features : list(arraylike)
        For each sequence input, we should have predictions or scores derived
        from those predictions across all the genomic/sequence-level features
        our model can predict. The length of this list is the number of
        sequences inputted to the model and the length of each element
        (`arraylike`) in the list is the number of sequence-level features.
    info_cols : list(arraylike)
        Identifying information attached to each sequence entry. Each item
        in `info_cols` is the label information for each row that is written
        to the file. All values in an element of `info_cols` will be written
        to a separate .txt file.
    hdf5_handle : h5py._hl.files.File
        File handle we use to write the data to the HDF5 file.
    info_handle : _io.TextIOWrapper
        File handle to write the elements of `info_cols` to a tab-separated
        .txt file.

    Returns
    -------
    int
    The updated start_index.
    """
    with open(info_filepath, 'a') as info_handle:
        for info_batch in info_cols:
            for info in info_batch:
                info_str = '\t'.join([str(i) for i in info])
                info_handle.write("{0}\n".format(info_str))
    with h5py.File(hdf5_filepath, 'a') as hdf5_handle:
        data = hdf5_handle["data"]
        for data_batch in data_across_features:
            data[start_index  : (start_index + data_batch.shape[0])] = data_batch
            start_index = start_index + data_batch.shape[0]

    return start_index

def write_NAs_to_file(info_cols, column_names, output_path):
    """
    Writes samples with NA predictions or scores to a tab-delimited file.

    Parameters
    ----------
    info_cols : list(arraylike)
        NA entries should still have identifying information attached
        to each entry. Each item in `info_cols` corresponds to each
        row that will be written to the file. The item then, is the list
        of column value that go in this row.
    column_names : list(str)
        Column names written as the first line of the file.
    output_path : str
        Path to the file where we should write the `NA`'s.

    Returns
    -------
    None
        Writes information to file
    """
    with open(output_path, 'w+') as file_handle:
        file_handle.write("{columns}\n".format(
            columns='\t'.join(column_names)))
        for info in info_cols:
            write_info = '\t'.join([str(i) for i in info])
            file_handle.write("{0}\n".format(write_info))


def probabilities_to_string(probabilities):
    """
    Converts a list of probability values (`float`s) to a list of
    `str` probability values, where each value is represented in
    scientific notation with 2 digits after the decimal.

    Parameters
    ----------
    probabilities : list(float)

    Returns
    -------
    list(str)

    """
    return ["{:.2e}".format(p) for p in probabilities]


class PredictionsHandler(metaclass=ABCMeta):
    """
    The abstract base class for handlers, which "handle" model
    predictions. Handlers are responsible for accepting predictions,
    storing these predictions or scores derived from the predictions,
    and then returning them in a user-specified output format (Selene
    currently supports TSV and HDF5 file outputs)

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
        a handler-specific name to the end of the path/prefix.
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
        self.needs_base_pred = False
        self._results = []
        self._samples = []
        self._NA_samples = []

        self._features = features
        self._columns_for_ids = columns_for_ids
        self._output_path_prefix = output_path_prefix
        self._output_format = output_format
        self._output_size = output_size
        if output_format == 'hdf5' and output_size is None:
            raise ValueError("output_size must be specified when output_format is hdf5.")

        self._output_filepath = None
        self._labels_filepath = None

        self._write_mem_limit = write_mem_limit

    def _create_write_handler(self, handler_filename):
        """
        TODO

        """
        output_path, filename_prefix = os.path.split(
            self._output_path_prefix)
        if len(filename_prefix) > 0:
            handler_filename = "{0}_{1}".format(
                filename_prefix, handler_filename)
        scores_filepath = os.path.join(output_path, handler_filename)
        if self._output_format == "tsv":
            self._output_filepath = "{0}.tsv".format(scores_filepath)
            with open(self._output_filepath, 'w+') as output_handle:
                column_names = self._columns_for_ids + self._features
                output_handle.write("{0}\n".format(
                    '\t'.join(column_names)))
        elif self._output_format == "hdf5":
            self._output_filepath = "{0}.h5".format(scores_filepath)
            with h5py.File(self._output_filepath, 'w') as output_handle:
                output_handle.create_dataset(
                    "data",
                    (self._output_size, len(self._features)),
                    dtype='float64')
            self._hdf5_start_index = 0

            labels_filename = "row_labels.txt"
            if len(filename_prefix) > 0:
                labels_filename = "{0}_{1}".format(
                    filename_prefix, labels_filename)
            self._labels_filepath = os.path.join(output_path, labels_filename)
            open(self._labels_filepath, 'w+')

    def _write_NAs_to_file(self,
                           output_path_prefix,
                           column_names):
        if self._NA_samples:
            output_path, prefix = os.path.split(output_path_prefix)
            NA_filename = "predictions.NA"
            if len(prefix) > 0:
                if '.ref' in prefix:
                    prefix, _ = prefix.split('.')
                NA_filename = "{0}_{1}".format(prefix, NA_filename)
            write_NAs_to_file(self._NA_samples,
                              column_names,
                              os.path.join(output_path, NA_filename))
            self._NA_samples = []

    def _reached_mem_limit(self):
        mem_used = (self._results[0].nbytes * len(self._results) +
                    getsizeof(self._samples[0]) * len(self._samples))
        if len(self._NA_samples) > 0:
            mem_used += getsizeof(self._NA_samples[0]) * len(self._NA_samples)
        return mem_used / 10**6 >= self._write_mem_limit

    def handle_NA(self, row_ids):
        """
        Handle rows without data. Will store the identifiers where predictions
        were not available in a separate list.

        Parameters
        ----------
        row_ids : arraylike(str)
            The list of row IDs

        """
        self._NA_samples.append(row_ids)

    @abstractmethod
    def handle_batch_predictions(self, *args, **kwargs):
        """
        Must be able to handle a batch of model predictions.
        """
        raise NotImplementedError

    def write_to_file(self):
        """
        Writes accumulated handler results to file.

        """
        self._write_NAs_to_file(
            self._output_path_prefix, self._columns_for_ids)

        if not self._results:
            return None
        if self._labels_filepath is not None:
            self._hdf5_start_index = write_to_hdf5_file(
                self._results,
                self._samples,
                self._output_filepath,
                self._labels_filepath,
                self._hdf5_start_index)
        else:
            write_to_tsv_file(self._results,
                              self._samples,
                              self._output_filepath)
        self._results = []
        self._samples = []
