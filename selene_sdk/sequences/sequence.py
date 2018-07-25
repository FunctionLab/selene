"""
This module provides the `Sequence` class. This class is the abstract
base class for biological sequence collections (e.g. genomes).

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
        The input sequence of length :math:`L`.
    base_to_index : dict
        A dict that maps input characters to indices, where the indices
        specify the column to assign as 1 when a base exists at the
        current position in the input. If a base does not exist at the
        current position in the input, it's corresponding column in the
        encoding is set as zero. Note that the rows correspond directly
        to the positions in the input sequence. For instance, with a
        a genome you would have each of `['A', 'C', 'G', 'T']` as keys,
        mapping to values of `[0, 1, 2, 3]`.
    bases_arr : list(str)
        The characters in the sequence's alphabet.

    Returns
    -------
    numpy.ndarray, dtype=numpy.float32
        The :math:`L \\times N` encoding of the sequence, where
        :math:`L` is the length of the input sequence and :math:`N` is
        the size of the sequence alphabet.

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
    encoding : numpy.ndarray, dtype=numpy.float32
        The :math:`L \\times N` encoding of the sequence, where
        :math:`L` is the length of the sequence, and :math:`N` is the
        size of the sequence alphabet.
    bases_arr : list(str)
        A list of the bases in the sequence's alphabet that corresponds
        to the correct columns for those bases in the encoding.
    unk_base : str
        The base corresponding to the "unknown" character in this
        encoding. See `selene_sdk.sequences.Sequence.UNK_BASE` for more
        information.

    Returns
    -------
    str
        The sequence of :math:`L` characters decoded from the
        input array.

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
                         complementary_base_dict):
    """
    The Genome DNA bases encoding is created such that the reverse
    encoding can be quickly computed.

    Parameters
    ----------
    encoding : numpy.ndarray
    bases_arr : list(str)
    base_to_index : dict
    complementary_base_dict : dict

    Returns
    -------
    numpy.ndarray

    """

    reverse_encoding = np.zeros(encoding.shape)
    for index, row in enumerate(encoding):
        base_pos = _get_base_index(row)
        if base_pos == -1:
            reverse_encoding[index, :] = 1 / len(bases_arr)
        else:
            base = complementary_base_dict[bases_arr[base_pos]]
            complem_base_pos = base_to_index[base]
            rev_index = encoding.shape[0] - row - 1
            reverse_encoding[rev_index, complem_base_pos] = 1
    return reverse_encoding


def reverse_complement_sequence(sequence, complementary_base_dict):
    """
    Finds the reverse complement of a sequence.

    Parameters
    ----------
    sequence : str
        The sequence to reverse complement.
    complementary_base_dict: dict
        A dict that maps bases (`str`) to their complementary bases
        (`str`).
    Returns
    -------
    str
        The reverse complement of the input sequence.

    """
    rev_comp_bases = [complementary_base_dict[b] for b in
                      sequence[::-1]]
    return ''.join(rev_comp_bases)


class Sequence(metaclass=ABCMeta):
    """
    The abstract base class for biological sequence classes.
    """
    @property
    @abstractmethod
    def BASE_TO_INDEX(self):
        """
        A dictionary mapping members of the alphabet (i.e. all
        possible symbols that can occur in a sequence) to integers.

        Returns
        -------
        dict
            The dictionary mapping the alphabet to integers.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def INDEX_TO_BASE(self):
        """
        A dictionary mapping integers to members of the alphabet (i.e.
        all possible symbols that can occur in a sequence). We expect
        that `INDEX_TO_BASE[i]==BASES_ARR[i]` is `True` for all
        valid `i`.

        Returns
        -------
        dict
            The dictionary mapping integers to the alphabet.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def BASES_ARR(self):
        """
        This is an array with the alphabet (i.e. all possible symbols
        that may occur in a sequence). We expect that
        `INDEX_TO_BASE[i]==BASES_ARR[i]` is `True` for all valid `i`.

        Returns
        -------
        numpy.ndarray, dtype=str
            The array of all members of the alphabet.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def UNK_BASE(self):
        """
        This is a base used to represent unknown positions. This is not
        the same as a character from outside the sequence's alphabet. A
        character from outside the alphabet is an error. A position with
        an unknown base signifies that the position is one of the bases
        from the alphabet, but we are uncertain which.

        Returns
        -------
        str
            The character representing an unknown base.

        """
        raise NotImplementedError()


    @abstractmethod
    def coords_in_bounds(self, *args, **kwargs):
        """Checks if queried coordinates are valid.

        Returns
        -------
        bool
            `True` if the coordinates are in bounds, otherwise `False`.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_sequence_from_coords(self, *args, **kwargs):
        """Extracts a string representation of a sequence at the
        given coordinates.

        Returns
        -------
        str
            The sequence of bases occuring at the queried coordinates.
            This sequence will be of length :math:`L` normally, but
            only if the coordinates are valid. Behavior is undefined for
            invalid coordinates.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_encoding_from_coords(self, *args, **kwargs):
        """Extracts the numerical encoding for a sequence occurring at
        the given coordinates.

        Returns
        -------
        numpy.ndarray, dtype=numpy.float32
            The :math:`L \\times N` encoding of the sequence occuring
            at queried coordinates, where :math:`L` is the length of the
            sequence, and :math:`N` is the size of the sequence type's
            alphabet. Behavior is undefined for invalid coordinates.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def sequence_to_encoding(cls, sequence):
        """Transforms a biological sequence into a numerical
        representation.

        Parameters
        ----------
        sequence : str
            The input sequence of characters.

        Returns
        -------
        numpy.ndarray, dtype=numpy.float32
            The :math:`L \\times N` encoding of the sequence, where
            :math:`L` is the length of the sequence, and :math:`N` is
            the size of the sequence type's alphabet.

        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def encoding_to_sequence(cls, encoding):
        """Transforms the input numerical representation of a sequence
        into a string representation.

        Parameters
        ----------
        encoding : numpy.ndarray, dtype=numpy.float32
            The :math:`L \\times N` encoding of the sequence, where
            :math:`L` is the length of the sequence, and :math:`N` is
            the size of the sequence type's alphabet.

        Returns
        -------
        str
            The sequence of bases decoded from the input array. This
            sequence will be of length :math:`L`.

        """
        raise NotImplementedError()
