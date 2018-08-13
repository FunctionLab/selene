"""
This module provides the BedFileSampler class.
"""
import numpy as np

from .file_sampler import FileSampler

class BedFileSampler(FileSampler):
    """
    A sampler for which the dataset is loaded directly from a `*.bed` file.

    Parameters
    ----------
    filepath : str
        The path to the file to load the data from.
    reference_sequence : selene_sdk.sequences.Sequence
        A reference sequence from which to create examples.
    n_samples : int
        Number of lines in the file. (`wc -l <filepath>`)
    sequence_length : int or None, optional
        Default is None. If the coordinates of each sample in the BED file
        already account for the full sequence (that is,
        `end - start = sequence_length`), there is no need to specify
        the sequence length. If `sequence_length` is not None, the length
        of each sample will be checked to determine whether the sample
        coordinates need to be truncated or expanded to reach the
        sequence length specified in the model architecture.
    targets_avail : bool, optional
        Default is False. If `targets_avail`, assumes that it is the
        last column of the `*.bed` file. The last column should contain
        the indices, separated by semicolons, of features (classes) found
        within a given sample's coordinates (e.g. 0;1;45;60). This assumes
        that we are only looking for the absence/presence of each feature
        within the interval.
    n_features : int or None, optional
        Default is None. If `targets_avail` is True, must specify
        `n_features`, the total number of features (classes).

    Attributes
    ----------
    filepath : str
        The path to the file to load the data from.
    reference_sequence : selene_sdk.sequences.Sequence
        A reference sequence from which to create examples.
    n_samples : int
        Number of lines in the file. (`wc -l <filepath>`)
    sequence_length : int or None, optional
        Default is None. If the coordinates of each sample in the BED file
        already account for the full sequence (that is,
        `end - start = sequence_length`), there is no need to specify
        the sequence length. If `sequence_length` is not None, the length
        of each sample will be checked to determine whether the sample
        coordinates need to be truncated or expanded to reach the
        sequence length specified in the model architecture.
    targets_avail : bool
        If `targets_avail`, assumes that it is the last column of the `*.bed`
        file. The last column should contain the indices, separated by
        semicolons, of features (classes) found within a given sample's
        coordinates (e.g. 0;1;45;60). This assumes that we are only looking
        or the absence/presence of each feature within the interval.
    n_features : int or None
        If `targets_avail` is True, must specify
        `n_features`, the total number of features (classes).

    """

    def __init__(self,
                 filepath,
                 reference_sequence,
                 n_samples,
                 sequence_length=None,
                 targets_avail=False,
                 n_features=None):
        """
        Constructs a new `BedFileSampler` object.
        """
        super(BedFileSampler, self).__init__()
        self.filepath = filepath
        self._file_handle = open(self.filepath, 'r')
        self.reference_sequence = reference_sequence
        self.sequence_length = sequence_length
        self.targets_avail = targets_avail
        self.n_features = n_features
        self.n_samples = n_samples

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
        sequences = []
        targets = None
        if self.targets_avail:
            targets = []
        while len(sequences) < batch_size:
            line = self._file_handle.readline()
            if not line:
                # TODO: add functionality to shuffle the file if sampler
                # reaches the end of the file.
                self._file_handle.close()
                self._file_handle = open(self.filepath, 'r')
                line = self._file_handle.readline()
            cols = line.split('\t')
            chrom = cols[0]
            start = int(cols[1])
            end = int(cols[2])
            strand_side = None
            features = None

            if len(cols) == 5:
                strand_side = cols[3]
                features = cols[4].strip()
            elif len(cols) == 4 and self.targets_avail:
                features = cols[3].strip()
            elif len(cols) == 4:
                strand_side = cols[3].strip()

            # if strand_side is None, assume strandedness does not matter.
            # can change this to randomly selecting +/- later
            strand_side = '+'
            n = end - start
            if self.sequence_length and n < self.sequence_length:
                diff = (self.sequence_length - n) / 2
                pad_l = int(np.floor(diff))
                pad_r = int(np.ceil(diff))
                start = start - pad_l
                end = end + pad_r
            elif self.sequence_length and n > self.sequence_length:
                start = int((n - self.sequence_length) // 2)
                end = int(start + self.sequence_length)

            sequence = self.reference_sequence.get_encoding_from_coords(
                chrom, start, end, strand=strand_side)
            if sequence.shape[0] == 0:
                continue

            sequences.append(sequence)
            if self.targets_avail:
                tgts = np.zeros((self.n_features))
                features = [int(f) for f in features.split(';') if f]
                tgts[features] = 1
                targets.append(tgts.astype(float))

        sequences = np.array(sequences)
        if self.targets_avail:
            targets = np.array(targets)
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
        if not self.targets_avail:
            raise ValueError(
                "No targets are specified in the *.bed file. "
                "Please use `get_data` instead.")
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
        targets_mat = np.vstack(targets_mat).astype(int)
        return sequences_and_targets, targets_mat
