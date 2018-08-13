"""
This module provides the `MatFileSampler` class and its supporting
methods.
"""
import h5py
import numpy as np
import scipy.io

from .file_sampler import FileSampler


def load_mat_file(filepath, sequence_key, targets_key=None):
    """
    Loads data from a `*.mat` file or a `*.h5` file.

    Parameters
    ----------
    filepath : str
        The path to the file to load the data from.
    sequence_key : str
        The key for the sequences data matrix.
    targets_key : str, optional
        Default is None. The key for the targets data matrix.

    Returns
    -------
    sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
        A tuple containing the numeric representation of the
        sequence examples and their corresponding labels.
        If no `targets_key` is specified, `targets` returned
        is None. The shape of `sequences` will be
        :math:`S \\times N \\times L`, where :math:`S` is
        the total number of samples, :math:`N` is the
        size of the sequence type's alphabet, and :math:`L`
        is the sequence length.
        The shape of `targets` will be :math:`S \\times F`,
        where :math:`F` is the number of features.
    """
    try:  # see if we can load the file using scipy first
        mat = scipy.io.loadmat(filepath)
        targets = None
        if targets_key:
            targets = mat[targets_key]
        return (mat[sequence_key], targets)
    except (NotImplementedError, ValueError):
        mat = h5py.File(filepath, 'r')
        sequences = mat[sequence_key][()]
        targets = None
        if targets_key:
            targets = mat[targets_key][()]
        mat.close()
        sequences = np.transpose(sequences, (2, 1, 0))
        return (sequences, targets.T)


class MatFileSampler(FileSampler):
    """
    A sampler for which the dataset is loaded directly from a `*.mat` file.

    Parameters
    ----------
    filepath : str
        The path to the file to load the data from.
    sequence_key : str
        The key for the sequences data matrix.
    targets_key : str, optional
        Default is None. The key for the targets data matrix.
    random_seed : int, optional
        Default is 436. Sets the random seed for sampling.
    shuffle : bool, optional
        Default is True. Shuffle the order of the samples in the matrix
        before sampling from it.

    Attributes
    ----------
    n_samples : int
        The number of samples in the data matrix.
    """

    def __init__(self,
                 filepath,
                 sequence_key,
                 targets_key=None,
                 random_seed=436,
                 shuffle=True):
        """
        Constructs a new `MatFileSampler` object.
        """
        super(MatFileSampler, self).__init__()
        sequences_mat, targets_mat = load_mat_file(
            filepath, sequence_key, targets_key=targets_key)
        self._sample_seqs = sequences_mat
        self._sample_tgts = targets_mat

        self.n_samples = self._sample_seqs.shape[0]

        self._sample_indices = np.arange(
            self.n_samples).tolist()
        self._sample_next = 0

        self._shuffle = shuffle
        if self._shuffle:
            np.random.shuffle(self._sample_indices)

    def sample(self, batch_size=1):
        """
        Draws a mini-batch of examples and their corresponding
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
        sample_up_to = self._sample_next + batch_size
        use_indices = None
        if sample_up_to >= len(self._sample_indices):
            if self._shuffle:
                np.random.shuffle(self._sample_indices)
            self._sample_next = 0
            use_indices = self._sample_indices[:batch_size]
        else:
            use_indices = self._sample_indices[self._sample_next:sample_up_to]
        self._sample_next += batch_size

        sequences = np.transpose(
            self._sample_seqs[use_indices, :, :], (0, 2, 1)).astype(float)
        if self._sample_tgts is not None:
            targets = self._sample_tgts[use_indices, :].astype(float)
            return (sequences, targets)
        return sequences,

    def get_data(self, batch_size, n_samples=None):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int, optional
            Default is None. The total number of samples to retrieve.

        Returns
        -------
        sequences : list(np.ndarray)
            The list of sequences grouped into batches.
            An element in the `sequences` list is of
            the shape :math:`B \\times L \\times N`, where :math:`B`
            is `batch_size`, :math:`L` is the sequence length,
            and :math:`N` is the size of the sequence type's alphabet.
        """
        if not n_samples:
            n_samples = self.n_samples
        sequences = []

        count = batch_size
        while count < n_samples:
            seqs, = self.sample(batch_size=batch_size)
            sequences.append(seqs)
            count += batch_size
        remainder = batch_size - (count - n_samples)
        seqs, = self.sample(batch_size=remainder)
        sequences.append(seqs)
        return sequences

    def get_data_and_targets(self, batch_size, n_samples=None):
        """
        This method fetches a subset of the sequence data and
        targets from the sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int, optional
            Default is None. The total number of samples to retrieve.

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
        if self._sample_tgts is None:
            raise ValueError(
                "No targets matrix was specified during sampler "
                "initialization. Please use `get_data` instead.")
        if not n_samples:
            n_samples = self.n_samples
        sequences_and_targets = []
        targets_mat = []

        count = batch_size
        while count < n_samples:
            seqs, tgts = self.sample(batch_size=batch_size)
            sequences_and_targets.append((seqs, tgts))
            targets_mat.append(tgts)
            count += batch_size
        remainder = batch_size - (count - n_samples)
        seqs, tgts = self.sample(batch_size=remainder)
        sequences_and_targets.append((seqs, tgts))
        targets_mat.append(tgts)
        # TODO: should not assume targets are always integers
        targets_mat = np.vstack(targets_mat).astype(int)
        return sequences_and_targets, targets_mat
