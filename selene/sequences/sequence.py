"""
This class is the abstract base class for biological sequences to load training examples from.
"""
from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from ._sequence import _fast_sequence_to_encoding

def sequence_to_encoding(sequence, base_to_index):
    """Converts an input sequence to its one hot encoding.

    Parameters
    ----------
    sequence : str
        The input sequence of length N.
    base_to_index : dict
       each of ('A', 'C', 'G', 'T' or 'U') as keys -> index (0, 1, 2, 3),
       specify the position to assign 1/0 when a given base exists/does not
       exist at a given position in the sequence.

    Returns
    -------
    np.ndarray, dtype=float32
        The N-by-4 encoding of the sequence.
    """
    return _fast_sequence_to_encoding(sequence, base_to_index)

def _get_base_index(encoding_row):
    for index, val in enumerate(encoding_row):
        if val == 0.25:
            return -1
        elif val == 1:
            return index
    return -1

def encoding_to_sequence(encoding, bases_arr):
    """Converts a sequence one hot encoding to its string
    sequence.

    Parameters
    ----------
    encoding : np.ndarray, dtype=float32
    bases_arr : list
        each of ('A', 'C', 'G', 'T' or 'U') in the order that
        corresponds to the correct columns for those bases in the encoding.

    Returns
    -------
    str
    """
    sequence = []
    for row in encoding:
        base_pos = _get_base_index(row)
        if base_pos == -1:
            sequence.append('N')
        else:
            sequence.append(bases_arr[base_pos])
    return "".join(sequence)

def get_reverse_encoding(encoding,
                         bases_arr,
                         base_to_index,
                         complementary_base):
    reverse_encoding = np.zeros(encoding.shape)
    for index, row in enumerate(encoding):
        base_pos = _get_base_index(row)
        if base_pos == -1:
            reverse_encoding[index, :] = 0.25
        else:
            base = complementary_base[bases_arr[base_pos]]
            complem_base_pos = base_to_index[base]
            reverse_encoding[index, complem_base_pos] = 1
    return reverse_encoding


class Sequence(metaclass=ABCMeta):
    """
    The base class for biological sequence classes.
    """
    @abstractmethod
    def sequence_in_bounds(self, *args, **kwargs):
        """
        Checks if given coordinates are in the sequence.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sequence_from_coords(self, *args, **kwargs):
        """
        Extracts a string of sequence at the given coordinates.
        """
        raise NotImplementedError

    @abstractmethod
    def get_encoding_from_coords(self, *args, **kwargs):
        """
        Extracts the numerical encoding for a sequence occuring at the given coordinates.
        """
        raise NotImplementedError

    @abstractmethod
    def sequence_to_encoding(self, sequence):
        """
        Transforms a biological sequence into a numerical representation.
        """
        raise NotImplementedError

    @abstractmethod
    def encoding_to_sequence(self, encoding):
        """
        Transforms the input numerical representation of a biological sequence into a string representation.
        """
        raise NotImplementedError
