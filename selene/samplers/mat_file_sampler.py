"""
This module provides the `MatFileSampler` class and its supporting
methods.
"""
import h5py
import numpy as np
import scipy.io

from .sampler import Sampler


class MatFileSampler(Sampler):
    """
    A sampler for which the training/validation/test data are loaded
    directly from a `*.mat` file.

    Parameters
    ----------
    training_data_path : str
        Path to the mat file containing the training data.
    validation_data_path : str
        Path to the mat file containing the validation data.
    test_data_path : str or None, optional
        Default is `None`. The path to the mat file containing the test
        data. If `None`, no test data will be used.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    mode : str, optional
        Default is `"train"`.  The mode to run the sampler in.

    Attributes
    ----------
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.

    """
    def __init__(self,
                 training_data_path,
                 validation_data_path,
                 test_data_path=None,
                 seed=436,
                 mode="train"):
        """
        Constructs a new `MatFileSampler` object.
        """
        super(MatFileSampler, self).__init__(seed=seed)
        if test_data_path:
            self.modes.append("test")

        self._sample_from_mode = {}
        self._randcache = {}
        for mode in self.modes:
            self._sample_from_mode[mode] = {
                "data": None,
                "indices": [],
                "sample_next": 0
            }

        train_data_mats, n_train = self._load_mat_file(training_data_path)
        self._sample_from_mode["train"]["data"] = train_data_mats
        self._sample_from_mode["train"]["indices"] = np.arange(n_train).tolist()

        valid_data_mats, n_validate = self._load_mat_file(validation_data_path)
        self._sample_from_mode["validate"]["data"] = valid_data_mats
        self._sample_from_mode["validate"]["indices"] = \
            np.arange(n_validate).tolist()

        if test_data_path:
            test_data_mats, n_test = self._load_mat_file(test_data_path)
            self._sample_from_mode["test"]["data"] = test_data_mats
            self._sample_from_mode["test"]["indices"] = \
                np.arange(n_test).tolist()

        self.sequence_length = train_data_mats[0].shape[1]
        self.n_features = train_data_mats[1].shape[1]

        for mode, mode_info in self._sample_from_mode.items():
            np.random.shuffle(mode_info["indices"])

    def _load_mat_file(self, mode, input_path,
                       data_key_suffix="xdata",
                       target_key_suffix="data"):
        """
        Loads data from a `*.mat` file.
        Assumes the keys are "<mode>xdata" and "<mode>data",
        e.g. "trainxdata" being the matrix of one-hot encoded input sequences
        and "traindata" being the matrix of corresponding genomic features.

        Parameters
        ----------
        mode : str
            The mode that these samples should be used for. See
            `selene.samplers.MatFileSampler.modes` for more information.
        input_path : str
            The path to the mat file to load the data from.
        data_key_suffix : str
            The suffix that will be appended to the run mode string to
            form a key. This key is then used to index into the mat file
            that maps these keys to the dataset of examples for the
            specified mode.
        target_key_suffix : str
            The suffix that will be appended to the run mode string to
            form a key. This key is then used to index into the mat file
            that maps these keys to the dataset of labels for the
            specified mode.

        """
        seqs_key = "{0}{1}".format(mode, data_key_suffix)
        tgts_key = "{0}{1}".format(mode, target_key_suffix)
        try:  # see if it will load using scipy first
            mat = scipy.io.loadmat(input_path)
            n_samples, _ = mat[tgts_key].shape
            return (mat[seqs_key], mat[tgts_key]), n_samples
        except NotImplementedError:
            mat_fh = h5py.File(input_path, 'r')
            sequence_data = mat[seqs_key][()].T
            tgts = mat[tgts_key][()].T
            n_samples, _ = tgts.shape
            mat_fh.close()
            return (sequence_data, tgts, n_samples)

    def sample(self, batch_size=1):
        """
        Randomly draws a mini-batch of examples and their corresponding
        labels.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The number of examples to include in the
            mini-batch.

        Returns
        -------
        sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence examples and their corresponding labels. The
            shape of `sequences` will be
            :math:`B \\times L \\times N`, where :math:`B` is
            `batch_size`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` will be :math:`B \\times F`,
            where :math:`F` is the number of features.

        """
        mode = self.mode
        indices = self._sample_from_mode[mode]["indices"]
        input_data, target_data = self._sample_from_mode[mode]["data"]

        sample_up_to = self._sample_from_mode["sample_next"] + batch_size
        use_indices = None
        if sample_up_to > len(indices):
            np.random.shuffle(self._sample_from_mode[mode]["indices"])
            self._sample_from_mode["sample_next"] = 0
            use_indices = indices[:batch_size]
        else:
            sample_next = self._sample_from_mode["sample_next"] + batch_size
            use_indices = indices[sample_next:sample_next + batch_size]
        self._sample_from_mode["sample_next"] += batch_size
        sequences = np.transpose(
            input_data[use_indices, :, :], (0, 2, 1)).astype(float)
        targets = target_data[use_indices, :].astype(float)
        return (sequences, targets)

    def get_data_and_targets(self, mode, batch_size, n_samples):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches. This method also allows the user to
        specify what operating mode to run the sampler in when fetching
        the data.

        Parameters
        ----------
        mode : str
            The mode to run the sampler in when fetching the samples.
            See `selene.samplers.MatFileSampler.modes` for more
            information.
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int
            The total number of samples to retrieve.

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
        indices = self._sample_from_mode[mode]["indices"]
        input_data, target_data = self._sample_from_mode[mode]["data"]

        sequences_and_targets = []
        targets_mat = []
        for _ in range(0, n_samples, self.batch_size):
            sample_next = self._sample_from_mode["sample_next"] + batch_size
            use_indices = None
            if sample_next > len(indices):
                np.random.shuffle(self._sample_from_mode[mode]["indices"])
                self._sample_from_mode["sample_next"] = 0
                use_indices = indices[:batch_size]
            else:
                use_indices = indices[sample_next:sample_next + batch_size]
            self._sample_from_mode["sample_next"] += batch_size
            sequences = np.transpose(
                input_data[use_indices, :, :], (0, 2, 1)).astype(float)
            targets = target_data[use_indices, :].astype(float)

            sequences_and_targets.append((sequences, targets))
            targets_mat.append(targets)
        targets_mat = np.vstack(targets_mat)
        return sequences_and_targets, targets_mat

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
            to retrieve. If `None`, it will retrieve all validation
            data.

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
        if not n_samples:
            n_samples = len(self._sample_from_mode["validate"].indices)
        return self.get_data_and_targets("validate", batch_size, n_samples)
