"""This module provides the `Genome` class. This class wraps the indexed
FASTA file for an organism's genomic sequence. It supports retrieving
parts of the sequence and converting these parts into their one-hot
encodings.

"""
import numpy as np
import pyfaidx

from .sequence import Sequence
from .sequence import sequence_to_encoding
from .sequence import encoding_to_sequence


def _get_sequence_from_coords(len_chrs, genome_sequence, chrom,
                              start, end, strand='+'):
    """Gets the genomic sequence at the input coordinates.

    Parameters
    ----------
    len_chrs : dict
        A dictionary mapping chromosome names to lengths.
    genome_sequence : function
        A closure that extracts a sequence from a genome.
    chrom : str
        The name of the chromosomes, e.g. "chr1".
    start : int
        The 0-based start coordinate of the sequence.
    end : int
        One past the last coordinate of the sequence.
    strand : {'+', '-'}, optional
        Default is '+'. The strand the sequence is located on.

    Returns
    -------
    str
        The genomic sequence occurring at the input coordinates.

    Raises
    ------
    ValueError
        If the input char to `strand` is not one of the specified
        choices.

    """
    if start > len_chrs[chrom] or end > (len_chrs[chrom] + 1) \
            or start < 0:
        return ""

    if strand == '+' or strand == '-':
        return genome_sequence(chrom, start, end, strand)
    else:
        raise ValueError(
            "Strand must be one of '+' or '-'. Input was {0}".format(
                strand))


class Genome(Sequence):
    """This class provides access to an organism's genomic sequence.

    This class supports retrieving parts of the sequence and converting
    these parts into their one-hot encodings. It is essentially a
    wrapper class around the `pyfaix.Fasta` class.

    Attributes
    ----------
    genome : pyfaidx.Fasta
        The Fasta file containing the genome sequence.
    chrs : list(str)
        The list of chromosome names.
    len_chrs : dict
        The length of each chromosome sequence in the file.

    """

    BASES_ARR = np.array(['A', 'C', 'G', 'T'])
    INDEX_TO_BASE = {
        0: 'A', 1: 'C', 2: 'G', 3: 'T'
    }
    BASE_TO_INDEX = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
        'a': 0, 'c': 1, 'g': 2, 't': 3,
    }  # TODO: Consider renaming BASE to ALPHA or CHAR to make more general?
    COMPLEMENTARY_BASE = {
        'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
        'a': 'T', 'c': 'G', 'g': 'C', 't': 'A', 'n': 'N'
    }
    UNK_BASE = "N"

    def __init__(self, input_path):
        """Constructs a `Genome` object.

        Parameters
        ----------
        input_path : str
            Path to an indexed FASTA file, that is, a *.fasta file with
            a corresponding *.fai file in the same directory. This file
            should contain the target organism's genome sequence.

        """
        self.genome = pyfaidx.Fasta(input_path)
        self.chrs = sorted(self.genome.keys())
        self.len_chrs = self._get_len_chrs()

    def get_chrs(self):
        """Gets the list of chromosome names.

        Returns
        -------
        list(str)
            A list of the chromosome names.

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
        """Check if the region we want to query is within the bounds of
         the queried chromosome.

        Parameters
        ----------
        chrom : str
            The name of the chromosomes, e.g. "chr1".
        start : int
            The 0-based start coordinate of the sequence.
        end : int
            One past the last coordinate of the sequence.

        Returns
        -------
        bool
            Whether we can retrieve a sequence from the bounds specified
            in the input.

        """
        if chrom not in self.len_chrs:
            return False
        if (start > self.len_chrs[chrom] or end > (self.len_chrs[chrom] + 1)
                or start < 0):
            return False
        return True

    def get_sequence_from_coords(self, chrom, start, end, strand='+'):
        """Gets the queried chromosome's sequence at the input
         coordinates.

        Parameters
        ----------
        chrom : str
            The name of the chromosomes, e.g. "chr1".
        start : int
            The 0-based start coordinate of the sequence.
        end : int
            One past the last coordinate of the sequence.
        strand : {'+', '-'}, optional
            Default is '+'. The strand the sequence is located on.

        Returns
        -------
        str
            The genomic sequence.

        Raises
        ------
        ValueError
            If the input char to `strand` is not one of the specified
            choices.

        """
        return _get_sequence_from_coords(self.len_chrs, self._genome_sequence,
                                         chrom, start, end, strand)

    def get_encoding_from_coords(self, chrom, start, end, strand='+'):
        """Gets the one-hot encoding of the genomic sequence at the
        queried coordinates.

        Parameters
        ----------
        chrom : str
            The name of the chromosome or region, e.g. "chr1".
        start : int
            The 0-based start coordinate of the first position in the
            sequence.
        end : int
            One past the 0-based last position in the sequence.
        strand : {'+', '-'}, optional
            Default is '+'. The strand the sequence is located on.

        Returns
        -------
        numpy.ndarray, dtype=bool
            The N-by-4 encoding of the sequence.

        Raises
        ------
        ValueError
            If the input char to `strand` is not one of the specified
            choices.
            (Raised in the call to `self.get_sequence_from_coords`)

        """
        sequence = self.get_sequence_from_coords(chrom, start, end, strand)
        encoding = self.sequence_to_encoding(sequence)
        return encoding

    @classmethod
    def sequence_to_encoding(cls, sequence):
        """Converts an input sequence to its one-hot encoding.

        Parameters
        ----------
        sequence : str
            A nucleotide sequence of length N.

        Returns
        -------
        numpy.ndarray, dtype=numpy.float32
            The N-by-4 one-hot encoding of the sequence.

        """
        return sequence_to_encoding(sequence, cls.BASE_TO_INDEX, cls.BASES_ARR)

    @classmethod
    def encoding_to_sequence(cls, encoding):
        """Converts an input one-hot encoding to its DNA sequence.

        Parameters
        ----------
        encoding : numpy.ndarray, dtype=numpy.float32
            An N-by-4 one-hot encoding of the sequence, where N is the
            length of the output sequence.

        Returns
        -------
        str
            The sequence of N nucleotides decoded from the input array.

        """
        return encoding_to_sequence(encoding, cls.BASES_ARR, cls.UNK_BASE)
