"""This module provides the `Sequence` class. This class class is the base class for biological
sequence collections (e.g. genomes).

"""
from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from ._sequence import _fast_sequence_to_encoding


def sequence_to_encoding(sequence, base_to_index, bases_arr):
    """Converts an input sequence to its one-hot encoding.

    Parameters
    ----------
    sequence : str
        The input sequence of length N.
    base_to_index : dict
        A dict that maps input characters to indices, where the indices specify the position to assign 1/0 to
        when a base exists/does not exist at a given position in the sequence. For instance, for a genome you would
        have each of `['A', 'C', 'G', 'T']` as keys, mapping to values of `[0, 1, 2, 3]`.
    bases_arr : list(str)
        The characters in the sequence's alphabet.

    Returns
    -------
    np.ndarray, dtype=numpy.float32
        The N-by-X encoding of the sequence, where X is the size of the alphabet.'
    """
    return _fast_sequence_to_encoding(sequence, base_to_index, len(bases_arr))


def _get_base_index(encoding_row):
    unk_val = 1 / len(encoding_row)
    for index, val in enumerate(encoding_row):
        if np.isclose(val, unk_val) is True:
            return -1
        elif val == 1:
            return index
    return -1


def encoding_to_sequence(encoding, bases_arr, unk_base):
    """Converts a sequence one-hot encoding to its string sequence.

    Parameters
    ----------
    encoding : np.ndarray, dtype=numpy.float32
        The N-by-X encoding of the sequence, where N is the length and X is the size of the alphabet.
    bases_arr : list(str)
        A list of the bases in the sequence's alphabet that corresponds to the
        correct columns for those bases in the encoding.
    unk_base : str
        The base corresponding to the "unknown" character in this encoding.

    Returns
    -------
    str
        The sequence of N characters decoded from the input array.

    """
    sequence = []
    for row in encoding:
        base_pos = _get_base_index(row)
        if base_pos == -1:
            sequence.append(unk_base)
        else:
            sequence.append(bases_arr[base_pos])
    return "".join(sequence)


def get_reverse_encoding(encoding,
                         bases_arr,
                         base_to_index,
                         complementary_base):
    # TODO(DOCUMENTATION): What is the documentation for this?
    reverse_encoding = np.zeros(encoding.shape)
    for index, row in enumerate(encoding):
        base_pos = _get_base_index(row)
        if base_pos == -1:
            reverse_encoding[index, :] = 1 / len(bases_arr)
        else:
            base = complementary_base[bases_arr[base_pos]]
            complem_base_pos = base_to_index[base]
            rev_index = encoding.shape[0] - row - 1
            reverse_encoding[rev_index, complem_base_pos] = 1
    return reverse_encoding


class Sequence(metaclass=ABCMeta):
    """The base class for biological sequence classes.

    """
    BASE_TO_INDEX = None  # TODO: Determine if this is a good way to specify these requirements.
    INDEX_TO_BASE = None
    BASES_ARR = None
    UNK_BASE = '?'

    @abstractmethod
    def sequence_in_bounds(self, *args, **kwargs):
        """Checks if given coordinates are in the sequence.

        """
        raise NotImplementedError()

    @abstractmethod
    def get_sequence_from_coords(self, *args, **kwargs):
        """Extracts a string of sequence at the given coordinates.

        """
        raise NotImplementedError()

    @abstractmethod
    def get_encoding_from_coords(self, *args, **kwargs):
        """Extracts the numerical encoding for a sequence occuring at the given coordinates.

        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def sequence_to_encoding(cls, sequence):
        """Transforms a biological sequence into a numerical representation.

        Parameters
        ----------
        sequence : str
            The input sequence of characters.

        Returns
        -------
        numpy.ndarray, dtype=numpy.float32
            The N-by-X encoding of the sequence, where N is the length of the input sequence and
            X is the size of the sequence type alphabet.

        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def encoding_to_sequence(cls, encoding):
        """Transforms the input numerical representation of a sequence into a string representation.

        Parameters
        ----------
        encoding : numpy.ndarray, dtype=numpy.float32
            The N-by-X encoding of the sequence, where X is the size of the sequence type alphabet.

        Returns
        -------
        str
            The sequence of N bases decoded from the input array.

        """
        raise NotImplementedError()
