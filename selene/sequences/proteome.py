"""This module provides the `Proteome` clasee. This class wraps the
indexed FASTA file for an organism's proteomic sequence. It supports
retrieving parts of the sequence and converting these parts into their
one-hot encodings.

"""
import numpy as np
import pyfaidx

from .sequence import Sequence
from .sequence import sequence_to_encoding
from .sequence import encoding_to_sequence


def _get_sequence_from_coords(len_prots, proteome_sequence,
                              prot, start, end):
    """Gets the amino acid sequence at specified coordinates.

    Parameters
    ----------
    len_prots : dict
        A dictionary mapping protein names to lengths.
    proteome_sequence : function
        A closure that returns the sequence at given coordinates.
    prot : str
        The name of a protein, e.g. "YFP".
    start : int
        The 0-based start coordinate of the first position in the
        sequence.
    end : int
        One past the 0-based last position in the sequence.

    Returns
    -------
    str
        The amino acid sequence.

    """
    if start > len_prots[prot] or (end > len_prots[prot] + 1) or start < 0:
        return ""
    return proteome_sequence(prot, start, end)


class Proteome(Sequence):
    """Provides access to an organism's proteomic sequence.

    It supports retrieving parts of the sequence and converting these
    parts into their one-hot encodings. It is essentially a wrapper
    class around the `pyfaix.Fasta` class.

    Attributes
    ----------
    proteome : pyfaidx.Fasta
        The Fasta file containing the protein sequences.
    prots : list(str)
        The list of protein names.
    len_prots : dict
        The length of each protein sequence in the file.

    """

    BASES_ARR = np.array(['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                          'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
    INDEX_TO_BASE = {0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'E', 6: 'Q',
                     7: 'G', 8: 'H', 9: 'I', 10: 'L', 11: 'K', 12: 'M',
                     13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y',
                     19: 'V'}
    BASE_TO_INDEX = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6,
        'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13,
        'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }  # TODO: Consider renaming BASE to ALPHA or CHAR to make more general?
    UNK_BASE = "X"

    def __init__(self, input_path):
        """Constructs a `Proteome` object.

        Parameters
        ----------
        input_path : str
            Path to an indexed FASTA file containing amino acid
            sequences, that is, a *.faa file with a corresponding *.fai
            file in the same directory. File should contain the
            sequences from which training examples will be created.

        """
        self.proteome = pyfaidx.Fasta(input_path)
        self.prots = sorted(self.proteome.keys())
        self.len_prots = self._get_len_prots()

    def get_prots(self):
        """Gets the list of protein names.

        Returns
        -------
        list(str)
            The list of protein names.

        """
        return self.prots

    def get_prot_lens(self):
        """Gets the length of each protein sequence in the file.

        Returns
        -------
        list(tup)
            Tuples of protein name (str) and protein length (int).

        """
        return list(self.len_prots.items())

    def _get_len_prots(self):
        len_prots = {}
        for prot in self.prots:
            len_prots[prot] = len(self.proteome[prot])
        return len_prots

    def _proteome_sequence(self, prot, start, end):
        return self.proteome[prot][start:end].seq

    def sequence_in_bounds(self, prot, start, end):
        """Check if the region we want to query is within the bounds of
         the queried protein.

        Parameters
        ----------
        prot : str
            The name of the protein, e.g. "YFP".
        start : int
            The 0-based start coordinate of the first position in the
            sequence.
        end : int
            One past the 0-based last position in the sequence.

        Returns
        -------
        bool
            Whether we can retrieve a sequence from the bounds specified
            in the input.

        """
        if (start > self.len_prots[prot] or end > (self.len_prots[prot] + 1)
                or start < 0):
            return False
        return True

    def get_sequence_from_coords(self, prot, start, end):
        """Gets the queried protein sequence at the input coordinates.

        Parameters
        ----------
        prot : str
            The protein name, e.g. "YFP".
        start : int
            The 0-based start coordinate of the first position in the
            sequence.
        end : int
            One past the 0-based last position in the sequence.

        Returns
        -------
        str
            The amino acid sequence at the specified coordinates.

        """
        return _get_sequence_from_coords(
            self.len_prots, self._proteome_sequence, prot, start, end)

    def get_encoding_from_coords(self, prot, start, end):
        """Gets the one-hot encoding of the protein's sequence at the
        input coordinates.

        Parameters
        ----------
        prot : str
            The name of the protein, e.g. "YFP".
        start : int
            The 0-based start coordinate of the first position in the
             sequence.
        end : int
            One past the 0-based last position in the sequence.

        Returns
        -------
        numpy.ndarray, dtype=bool
            The N-by-20 encoding of the sequence.

        """
        sequence = self.get_sequence_from_coords(prot, start, end)
        encoding = self.sequence_to_encoding(sequence)
        return encoding

    @classmethod
    def sequence_to_encoding(cls, sequence):
        """Converts an input sequence to its one-hot encoding.

        Parameters
        ----------
        sequence : str
            The input sequence of amino acids of length N.

        Returns
        -------
        numpy.ndarray, dtype=numpy.float32
            The N-by-20 encoding of the sequence.

        """
        return sequence_to_encoding(sequence, cls.BASE_TO_INDEX, cls.BASES_ARR)

    @classmethod
    def encoding_to_sequence(cls, encoding):
        """Converts an input one-hot encoding to its amino acid sequence.

        Parameters
        ----------
        encoding : numpy.ndarray, dtype=numpy.float32
            The N-by-20 encoding of the sequence, where N is the length
            of the output sequence.

        Returns
        -------
        str
            The sequence of N amino acids decoded from the input array.

        """
        return encoding_to_sequence(encoding, cls.BASES_ARR, cls.UNK_BASE)
