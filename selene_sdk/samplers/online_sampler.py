"""
This module provides the `OnlineSampler` class and supporting methods.
Objects of the class `OnlineSampler`, are samplers which load examples
"on the fly" rather than storing them all persistently in memory.

"""
from abc import ABCMeta
import os
import random

import numpy as np

from .sampler import Sampler
from ..targets import GenomicFeatures


class OnlineSampler(Sampler, metaclass=ABCMeta):
    """
    A sampler in which training/validation/test data is constructed
    from random sampling of the dataset for each batch passed to the
    model. This form of sampling may alleviate the problem of loading an
    extremely large dataset into memory when developing a new model.

    Parameters
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        A reference sequence from which to create examples.
    target_path : str
        Path to tabix-indexed, compressed BED file (`*.bed.gz`) of genomic
        coordinates mapped to the genomic features we want to predict.
    features : list(str)
        List of distinct features that we aim to predict.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    validation_holdout : list(str) or float, optional
        Default is `['chr6', 'chr7']`. Holdout can be regional or
        proportional. If regional, expects a list (e.g. `['X', 'Y']`).
        Regions must match those specified in the first column of the
        tabix-indexed BED file. If proportional, specify a percentage
        between (0.0, 1.0). Typically 0.10 or 0.20.
    test_holdout : list(str) or float, optional
        Default is `['chr8', 'chr9']`. See documentation for
        `validation_holdout` for additional information.
    sequence_length : int, optional
        Default is 1000. Model is trained on sequences of `sequence_length`
        where genomic features are annotated to the center regions of
        these sequences.
    center_bin_to_predict : int, optional
        Default is 200. Query the tabix-indexed file for a region of
        length `center_bin_to_predict`.
    feature_thresholds : float [0.0, 1.0], optional
        Default is 0.5. The `feature_threshold` to pass to the
        `GenomicFeatures` object.
    mode : {'train', 'validate', 'test'}, optional
        Default is `'train'`. The mode to run the sampler in.
    save_datasets : list(str), optional
        Default is `[]` the empty list. The list of modes for which we should
        save the sampled data to file (e.g. `["test", "validate"]`).
    output_dir : str or None, optional
        Default is None. The path to the directory where we should
        save sampled examples for a mode. If `save_datasets` is
        a non-empty list, `output_dir` must be specified. If
        the path in `output_dir` does not exist it will be created
        automatically.

    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        The reference sequence that examples are created from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features that we
        would like to predict.
    validation_holdout : list(str) or float
        The samples to hold out for validating model performance. These
        can be "regional" or "proportional". If regional, this is a list
        of region names (e.g. `['chrX', 'chrY']`). These regions must
        match those specified in the first column of the tabix-indexed
        BED file. If proportional, this is the fraction of total samples
        that will be held out.
    test_holdout : list(str) or float
        The samples to hold out for testing model performance. See the
        documentation for `validation_holdout` for more details.
    sequence_length : int
        The length of the sequences to  train the model on.
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.

    Raises
    ------
    ValueError
            If `mode` is not a valid mode.
    ValueError
        If the parities of `sequence_length` and `center_bin_to_predict`
        are not the same.
    ValueError
        If `sequence_length` is smaller than `center_bin_to_predict` is.
    ValueError
        If the types of `validation_holdout` and `test_holdout` are not
        the same.

    """
    STRAND_SIDES = ('+', '-')
    """
    Defines the strands that features can be sampled from.
    """

    def __init__(self,
                 reference_sequence,
                 target_path,
                 features,
                 seed=436,
                 validation_holdout=['chr6', 'chr7'],
                 test_holdout=['chr8', 'chr9'],
                 sequence_length=1001,
                 center_bin_to_predict=201,
                 feature_thresholds=0.5,
                 mode="train",
                 save_datasets=[],
                 output_dir=None):

        """
        Creates a new `OnlineSampler` object.
        """
        super(OnlineSampler, self).__init__(
            features,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed + 1)

        if isinstance(center_bin_to_predict, int):
            if (sequence_length + center_bin_to_predict) % 2 != 0:
                raise ValueError(
                    "Sequence length of {0} with a center bin length of {1} "
                    "is invalid. These 2 inputs should both be odd or both be "
                    "even.".format(sequence_length, center_bin_to_predict))

        # specifying a test holdout partition is optional
        if test_holdout:
            self.modes.append("test")
            if isinstance(validation_holdout, (list,)) and \
                    isinstance(test_holdout, (list,)):
                self.validation_holdout = [
                    str(c) for c in validation_holdout]
                self.test_holdout = [str(c) for c in test_holdout]
                self._holdout_type = "chromosome"
            elif isinstance(validation_holdout, float) and \
                    isinstance(test_holdout, float):
                self.validation_holdout = validation_holdout
                self.test_holdout = test_holdout
                self._holdout_type = "proportion"
            else:
                raise ValueError(
                    "Validation holdout and test holdout must have the "
                    "same type (list or float) but validation was "
                    "type {0} and test was type {1}".format(
                        type(validation_holdout), type(test_holdout)))
        else:
            self.test_holdout = None
            if isinstance(validation_holdout, (list,)):
                self.validation_holdout = [
                    str(c) for c in validation_holdout]
                self._holdout_type = "chromosome"
            elif isinstance(validation_holdout, float):
                self.validation_holdout = validation_holdout
                self._holdout_type = "proportion"
            else:
                raise ValueError(
                    "Validation holdout must be of type list (chromosomal "
                    "holdout) or float (proportion holdout) but was type "
                    "{0}.".format(type(validation_holdout)))

        if mode not in self.modes:
            raise ValueError(
                "Mode must be one of {0}. Input was '{1}'.".format(
                    self.modes, mode))
        self.mode = mode

        self.sequence_length = sequence_length
        window_radius = int(self.sequence_length / 2)
        self._start_window_radius = window_radius
        self._end_window_radius = window_radius
        if self.sequence_length % 2 != 0:
            self._end_window_radius += 1

        if isinstance(center_bin_to_predict, int):
            bin_radius = int(center_bin_to_predict / 2)
            self._start_radius = bin_radius
            self._end_radius = bin_radius
            if center_bin_to_predict % 2 != 0:
                self._end_radius += 1
        else:
            if not isinstance(center_bin_to_predict, list) or \
                    len(center_bin_to_predict) != 2:
                raise ValueError(
                    "`center_bin_to_predict` needs to be either an int or a list of "
                    "two ints, but was type '{0}'".format(
                        type(center_bin_to_predict)))
            else:
                bin_start, bin_end = center_bin_to_predict
                if bin_start < 0 or bin_end > self.sequence_length:
                    ValueError(
                        "center_bin_to_predict [{0}, {1}]"
                        "is out-of-bound for sequence length {3}.".format(
                            bin_start, bin_end, self.sequence_length))
                self._start_radius = self._start_window_radius - bin_start
                self._end_radius = self._end_window_radius - (self.sequence_length - bin_end)

        self.reference_sequence = reference_sequence
        self.n_features = len(self._features)
        self.target = GenomicFeatures(
            target_path, self._features,
            feature_thresholds=feature_thresholds)
        self._save_filehandles = {}

    def get_feature_from_index(self, index):
        """
        Returns the feature corresponding to an index in the feature
        vector.

        Parameters
        ----------
        index : int
            The index of the feature to retrieve the name for.

        Returns
        -------
        str
            The name of the feature occurring at the specified index.
        """
        return self.target.index_feature_dict[index]

    def get_sequence_from_encoding(self, encoding):
        """
        Gets the string sequence from the one-hot encoding
        of the sequence.

        Parameters
        ----------
        encoding : numpy.ndarray
            An :math:`L \\times N` array (where :math:`L` is the length
            of the sequence and :math:`N` is the size of the sequence
            type's alphabet) containing the one-hot encoding of the
            sequence.

        Returns
        -------
        str
            The sequence of :math:`L` characters decoded from the input.
        """
        return self.reference_sequence.encoding_to_sequence(encoding)

    def save_dataset_to_file(self, mode, close_filehandle=False):
        """
        Save samples for each partition (i.e. train/validate/test) to
        disk.

        Parameters
        ----------
        mode : str
            Must be one of the modes specified in `save_datasets` during
            sampler initialization.
        close_filehandle : bool, optional
            Default is False. `close_filehandle=True` assumes that all
            data corresponding to the input `mode` has been saved to
            file and `save_dataset_to_file` will not be called with
            `mode` again.
        """
        if mode not in self._save_datasets:
            return
        samples = self._save_datasets[mode]
        if mode not in self._save_filehandles:
            self._save_filehandles[mode] = open(
                os.path.join(self._output_dir,
                             "{0}_data.bed".format(mode)),
                'w+')
        file_handle = self._save_filehandles[mode]
        while len(samples) > 0:
            cols = samples.pop(0)
            line = '\t'.join([str(c) for c in cols])
            file_handle.write("{0}\n".format(line))
        if close_filehandle:
            file_handle.close()

    def get_data_and_targets(self, batch_size, n_samples=None, mode=None):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches. This method also allows the user to
        specify what operating mode to run the sampler in when fetching
        the data.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is None. The total number of samples to retrieve.
            If `n_samples` is None and the mode is `validate`, will
            set `n_samples` to 32000; if the mode is `test`, will set
            `n_samples` to 640000 if it is None. If the mode is `train`
            you must have specified a value for `n_samples`.
        mode : str, optional
            Default is None. The mode to run the sampler in when
            fetching the samples. See
            `selene_sdk.samplers.IntervalsSampler.modes` for more
            information. If None, will use the current mode `self.mode`.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.

        """
        if mode is not None:
            self.set_mode(mode)
        else:
            mode = self.mode
        sequences_and_targets = []
        if n_samples is None and mode == "validate":
            n_samples = 32000
        elif n_samples is None and mode == "test":
            n_samples = 640000

        n_batches = int(n_samples / batch_size)
        for _ in range(n_batches):
            inputs, targets = self.sample(batch_size)
            sequences_and_targets.append((inputs, targets))
        targets_mat = np.vstack([t for (s, t) in sequences_and_targets])
        if mode in self._save_datasets:
            self.save_dataset_to_file(mode, close_filehandle=True)
        return sequences_and_targets, targets_mat

    def get_dataset_in_batches(self, mode, batch_size, n_samples=None):
        """
        This method returns a subset of the data for a specified run
        mode, divided into mini-batches.

        Parameters
        ----------
        mode : {'test', 'validate'}
            The mode to run the sampler in when fetching the samples.
            See `selene_sdk.samplers.IntervalsSampler.modes` for more
            information.
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of samples to retrieve.
            If `None`, it will retrieve 32000 samples if `mode` is validate
            or 640000 samples if `mode` is test or train.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            The list is length :math:`S`, where :math:`S =` `n_samples`.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`

        """
        return self.get_data_and_targets(
            batch_size, n_samples=n_samples, mode=mode)

    def get_validation_set(self, batch_size, n_samples=None):
        """
        This method returns a subset of validation data from the
        sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of validation examples
            to retrieve. If `None`, 32000 examples are retrieved.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.

        """
        return self.get_dataset_in_batches(
            "validate", batch_size, n_samples=n_samples)

    def get_test_set(self, batch_size, n_samples=None):
        """
        This method returns a subset of testing data from the
        sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of validation examples
            to retrieve. If `None`, 640000 examples are retrieved.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.


        Raises
        ------
        ValueError
            If no test partition of the data was specified during
            sampler initialization.
        """
        if "test" not in self.modes:
            raise ValueError("No test partition of the data was specified "
                             "during initialization. Cannot use method "
                             "`get_test_set`.")
        return self.get_dataset_in_batches("test", batch_size, n_samples)
