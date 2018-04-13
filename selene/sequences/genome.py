"""This class wraps the indexed FASTA file for an organism's genomic sequence.
It supports retrieving parts of the sequence and converting these parts
into their one hot encodings.
"""
import numpy as np
from pyfaidx import Fasta

from .sequence import Sequence
from ._genome import _fast_sequence_to_encoding


def _sequence_to_encoding(sequence, bases_encoding):
    """Converts an input sequence to its one hot encoding.

    Parameters
    ----------
    sequence : str
        The input sequence of length N.
    bases_encoding : dict
       each of ('A', 'C', 'G', 'T' or 'U') as keys -> index (0, 1, 2, 3),
       specify the position to assign 1/0 when a given base exists/does not
       exist at a given position in the sequence.

    Returns
    -------
    numpy.ndarray, dtype=bool
        The N-by-4 encoding of the sequence.
    """
    return _fast_sequence_to_encoding(sequence, bases_encoding)

def _get_base_index(encoding_row):
    for index, val in enumerate(encoding_row):
        if val == 0.25:
            return -1
        elif val == 1:
            return index
    return -1

def _encoding_to_sequence(encoding, bases_arr):
    sequence = []
    for row in encoding:
        base_pos = _get_base_index(row)
        if base_pos == -1:
            sequence.append('N')
        else:
            sequence.append(bases_arr[base_pos])
    return "".join(sequence)

def _get_sequence_from_coords(len_chrs, genome_sequence,
                              chrom, start, end, strand='+'):
    """Gets the genomic sequence given the chromosome, sequence start,
    sequence end, and strand side.

    Parameters
    ----------
    chrom : str
        e.g. "chr1".
    start : int
    end : int
    strand : {'+', '-'}, optional
        Default is '+'.

    Returns
    -------
    str
        The genomic sequence.

    Raises
    ------
    ValueError
        If the input char to `strand` is not one of the specified choices.
    """
    if start > len_chrs[chrom] or end > len_chrs[chrom] \
            or start < 0:
        return ""

    if strand == '+' or strand == '-':
        return genome_sequence(chrom, start, end, strand)
    else:
        raise ValueError(
            "Strand must be one of '+' or '-'. Input was {0}".format(
                strand))


class Genome(Sequence):

    BASES_ARR = np.array(['A', 'C', 'G', 'T'])
    BASES_DICT = dict(
        [(base, index) for index, base in enumerate(BASES_ARR)])

    def __init__(self, fa_file):
        """Wrapper class around the pyfaix.Fasta class.

        Parameters
        ----------
        fa_file : str
            Path to an indexed FASTA file, that is, a *.fasta file with a
            corresponding *.fai file in the same directory.
            File should contain the target organism's genome sequence.

        Attributes
        ----------
        genome : Fasta
        chrs : list[str]
        len_chrs : dict
            The length of each chromosome sequence in the file.
        """
        self.genome = Fasta(fa_file)
        self.chrs = sorted(self.genome.keys())
        self.len_chrs = self._get_len_chrs()

    def get_chrs(self):
        """Gets the list of chromosomes.

        Returns
        -------
        list(str)
        """
        return self.chrs

    def get_chr_lens(self):
        """Gets the length of each chromosome sequence in the file.

        Returns
        -------
        list(tup)
            Tuples of chromosome (str) and chromosome length (int).
        """
        return list(self.len_chrs.items())

    def _get_len_chrs(self):
        len_chrs = {}
        for chrom in self.chrs:
            len_chrs[chrom] = len(self.genome[chrom])
        return len_chrs

    def _genome_sequence(self, chrom, start, end, strand='+'):
        if strand == '+':
            return self.genome[chrom][start:end].seq
        else:
            return self.genome[chrom][start:end].reverse.complement.seq

    def sequence_in_bounds(self, chrom, start, end):
        """Check if the region we want to query is within the bounds of the
        start and end index for a chromosome in the genome.

        Parameters
        ----------
        chrom : str
            e.g. "chr1".
        start : int
        end : int

        Returns
        -------
        bool
            Whether we can retrieve a sequence from the bounds specified
            in the input
        """
        if start > self.len_chrs[chrom] or end > self.len_chrs[chrom] \
                or start < 0:
            return False
        return True

    def get_sequence_from_coords(self, chrom, start, end, strand='+'):
        """Gets the genomic sequence given the chromosome, sequence start,
        sequence end, and strand side.

        Parameters
        ----------
        chrom : str
            e.g. "chr1".
        start : int
        end : int
        strand : {'+', '-'}, optional
            Default is '+'.

        Returns
        -------
        str
            The genomic sequence.

        Raises
        ------
        ValueError
            If the input char to `strand` is not one of the specified choices.
        """
        return _get_sequence_from_coords(
            self.len_chrs, self._genome_sequence, chrom, start, end, strand)

    def get_encoding_from_coords(self, chrom, start, end, strand='+'):
        """Gets the genomic sequence given the chromosome, sequence start,
        sequence end, and strand side; and return its one hot encoding.

        Parameters
        ----------
        chrom : str
            e.g. "chr1".
        start : int
        end : int
        strand : {'+', '-'}, optional
            Default is '+'.

        Returns
        -------
        numpy.ndarray, dtype=bool
            The N-by-4 encoding of the sequence.

        Raises
        ------
        ValueError
            If the input char to `strand` is not one of the specified choices.
            (Raised in the call to `self.get_sequence_from_coords`)
        """
        sequence = self.get_sequence_from_coords(chrom, start, end, strand)
        encoding = self.sequence_to_encoding(sequence)
        return encoding

    def sequence_to_encoding(self, sequence):
        """Converts an input sequence to its one hot encoding.

        Parameters
        ----------
        sequence : str
            The input sequence of length N.

        Returns
        -------
        numpy.ndarray, dtype=float64
            The N-by-4 encoding of the sequence.
        """
        return _sequence_to_encoding(sequence, self.BASES_DICT)

    def encoding_to_sequence(self, encoding):
        """Converts an input encoding to its DNA sequence.

        Parameters
        ----------
        encoding : numpy.ndarray, dtype=float64
            The N-by-4 encoding of the sequence

        Returns
        -------
        str
        """
        return _encoding_to_sequence(encoding, self.BASES_ARR)
