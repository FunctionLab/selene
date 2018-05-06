"""This class wraps the indexed FASTA file for an organism's proteomic sequence.
    It supports retrieving parts of the sequence and converting these parts
    into their one hot encodings.
"""
import numpy as np
from pyfaidx import Fasta
from .sequence import Sequence
from .sequence import sequence_to_encoding
from .sequence import encoding_to_sequence


def _get_sequence_from_coords(len_prots, proteome_sequence,
                              prot, start, end):
    """Gets the amino acid sequence given the protein, sequence start, and
    sequence end.

    Parameters
    ----------
    prot : str
        e.g. "YFP".
    start : int
    end : int

    Returns
    -------
    str
        The amino acid sequence.

    """
    if start > len_prots[prot] or (end > len_prots[prot] + 1) or start < 0:
        return ""
    return proteome_sequence(prot, start, end)


class Proteome(Sequence):

    BASES_ARR = np.array(['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                          'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
    INDEX_TO_BASE = {0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C',  5: 'E', 6: 'Q',
                     7: 'G', 8: 'H', 9: 'I', 10: 'L', 11: 'K', 12: 'M', 13: 'F',
                     14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V'}
    BASE_TO_INDEX = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6,
        'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13,
        'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }  # TODO: Consider renaming BASE to ALPHA or CHAR to make it more general?
    UNK_BASE = "X"

    def __init__(self, faa_file):
        """Wrapper class around the pyfaix.Fasta class.

        Parameters
        ----------
        faa_file : str
            Path to an indexed FASTA file containing amino acid sequences,
            that is, a *.faa file with a corresponding *.fai file in the
            same directory. File should contain the sequences from which
            training examples will be created..

        Attributes
        ----------
        proteome : Fasta
        prots : list[str]
        len_prots : dict
            The length of each protein sequence in the file.
        """
        self.proteome = Fasta(faa_file)
        self.prots = sorted(self.proteome.keys())
        self.len_prots = self._get_len_prots()

    def get_prots(self):
        """Gets the list of proteins.

        Returns
        -------
        list(str)
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
        """Check if the region we want to query is within the bounds of the
        start and end index for a protein in the proteome.

        Parameters
        ----------
        prot : str
            e.g. "YFP".
        start : int
        end : int

        Returns
        -------
        bool
            Whether we can retrieve a sequence from the bounds specified
            in the input
        """
        if start > self.len_prots[prot] or end > (self.len_prots[prot] + 1) \
                or start < 0:
            return False
        return True

    def get_sequence_from_coords(self, prot, start, end):
        """Gets the amino acid sequence given the protein, sequence start, and
        sequence end.

        Parameters
        ----------
        prot : str
            e.g. "YFP".
        start : int
        end : int

        Returns
        -------
        str
            The amino acid sequence.

        """
        return _get_sequence_from_coords(
            self.len_prots, self._proteome_sequence, prot, start, end)

    def get_encoding_from_coords(self, prot, start, end):
        """Gets the proteomic sequence given the protein, sequence start,
        sequence end; and return its one hot encoding.

        Parameters
        ----------
        prot : str
            e.g. "YFP".
        start : int
        end : int

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
        """Converts an input sequence to its one hot encoding.

        Parameters
        ----------
        sequence : str
            The input sequence of length N.

        Returns
        -------
        numpy.ndarray, dtype=float64
            The N-by-20 encoding of the sequence.
        """
        return sequence_to_encoding(sequence, cls.BASE_TO_INDEX, cls.BASES_ARR)

    @classmethod
    def encoding_to_sequence(cls, encoding):
        """Converts an input encoding to its amino acid sequence.

        Parameters
        ----------
        encoding : numpy.ndarray, dtype=float64
            The N-by-20 encoding of the sequence

        Returns
        -------
        str
        """
        return encoding_to_sequence(encoding, cls.BASES_ARR, cls.UNK_BASE)
