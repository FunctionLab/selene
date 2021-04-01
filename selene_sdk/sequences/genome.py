"""
This module provides the `Genome` class. This class wraps the indexed
FASTA file for an organism's genomic sequence. It supports retrieving
parts of the sequence and converting these parts into their one-hot
encodings.

"""
import pkg_resources
import pyfaidx
import tabix

from functools import wraps
from .sequence import Sequence
from .sequence import sequence_to_encoding
from .sequence import encoding_to_sequence

def _not_blacklist_region(chrom, start, end, blacklist_tabix):
    """
    Check if the input coordinates are not overlapping with blacklist regions.

    Parameters
    ----------
    chrom : str
        The name of the chromosomes, e.g. "chr1".
    start : int
        The 0-based start coordinate of the sequence.
    end : int
        One past the last coordinate of the sequence.
    blacklist_tabix : tabix.open or None, optional
        Default is `None`. Tabix file handle if a file of blacklist regions
        is available.

    Returns
    -------
    bool
        False if the coordinates are overlaping with blacklist regions
        (if specified). Otherwise, return True.


    """
    if blacklist_tabix is not None:
        try:
            rows = blacklist_tabix.query(chrom, start, end)
            for row in rows:
                return False
        except tabix.TabixError:
            pass
    return True


def _check_coords(len_chrs,
                  chrom,
                  start,
                  end,
                  pad=False,
                  blacklist_tabix=None):
    """
    Check if the input coordinates are valid.

    Parameters
    ----------
    len_chrs : dict
        A dictionary mapping chromosome names to lengths.
    chrom : str
        The name of the chromosomes, e.g. "chr1".
    start : int
        The 0-based start coordinate of the sequence.
    end : int
        One past the last coordinate of the sequence.
    pad : bool, optional
        Default is `False`. Allow coordinates that are partially
        out of bounds.
    blacklist_tabix : tabix.open or None, optional
        Default is `None`. Tabix file handle if a file of blacklist regions
        is available.

    Returns
    -------
    bool
        True if the coordinates are valid (`start` and `end` are within
        chromosome boundaries and not overlaping with blacklist regions
        (if specified). Otherwise, return False.


    """
    return chrom in len_chrs and \
         start < len_chrs[chrom] and \
         start < end and \
         end > 0 and \
         (start >= 0 if not pad else True) and \
         (end <= len_chrs[chrom] if not pad else True) and \
         _not_blacklist_region(chrom, start, end, blacklist_tabix)



def _get_sequence_from_coords(len_chrs,
                              genome_sequence,
                              chrom,
                              start,
                              end,
                              strand='+',
                              pad=False,
                              blacklist_tabix=None):
    """
    Gets the genomic sequence at the input coordinates.

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
    strand : {'+', '-', '.'}, optional
        Default is '+'. The strand the sequence is located on. '.' is treated
        as '+'.
    pad : bool, optional
        Default is `False`. If the coordinates are out of bounds, make an
        in-bounds query and then pad the sequence to return the desired
        sequence length.
    blacklist_tabix : tabix.open or None, optional
        Default is `None`. Tabix file handle if a file of blacklist regions
        is available.

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

    if not _check_coords(len_chrs,
        chrom,
        start,
        end,
        pad=pad,
        blacklist_tabix=blacklist_tabix):
        return ""

    if strand != '+' and strand != '-' and strand != '.':
        raise ValueError(
            "Strand must be one of '+', '-', or '.'. Input was {0}".format(
                strand))

    end_pad = 0
    start_pad = 0
    if end > len_chrs[chrom]:
        end_pad = end - len_chrs[chrom]
        end = len_chrs[chrom]
    if start < 0:
        start_pad = -1 * start
        start = 0
    return (Genome.UNK_BASE * start_pad +
            genome_sequence(chrom, start, end, strand) +
            Genome.UNK_BASE * end_pad)


class Genome(Sequence):
    """This class provides access to an organism's genomic sequence.

    This class supports retrieving parts of the sequence and converting
    these parts into their one-hot encodings. It is essentially a
    wrapper class around the `pyfaidx.Fasta` class.

    Parameters
    ----------
    input_path : str
        Path to an indexed FASTA file, that is, a `*.fasta` file with
        a corresponding `*.fai` file in the same directory. This file
        should contain the target organism's genome sequence.
    blacklist_regions : str or None, optional
        Default is None. Path to a tabix-indexed list of regions from
        which we should not output sequences. This is used to ensure that
        we are not sampling from areas where we will never collect
        measurements. You can pass as input "hg19" or "hg38" to use the
        blacklist regions released by ENCODE. You can also pass in your own
        tabix-indexed .gz file.
    bases_order : list(str) or None, optional
        Default is None (use the default base ordering of
        `['A', 'C', 'G', 'T']`). Specify a different ordering of
        DNA bases for one-hot encoding.
    init_unpicklable : bool, optional
        Default is False. Delays initialization until a relevant method
        is called. This enables the object to be pickled after instantiation.
        `init_unpicklable` must be `False` when multi-processing is needed e.g.
        DataLoader. Set `init_unpicklable` to True if you are using this class
        directly through Selene's API and want to access class attributes
        without having to call on a specific method in Genome.

    Attributes
    ----------
    genome : pyfaidx.Fasta
        The FASTA file containing the genome sequence.
    chrs : list(str)
        The list of chromosome names.
    len_chrs : dict
        A dictionary mapping the names of each chromosome in the file to
        the length of said chromosome.

    """

    BASES_ARR = ['A', 'C', 'G', 'T']
    """
    This is an array with the alphabet (i.e. all possible symbols
    that may occur in a sequence). We expect that
    `INDEX_TO_BASE[i]==BASES_ARR[i]` is `True` for all valid `i`.

    """

    BASE_TO_INDEX = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
        'a': 0, 'c': 1, 'g': 2, 't': 3,
    }
    """
    A dictionary mapping members of the alphabet (i.e. all
    possible symbols that can occur in a sequence) to integers.
    """

    INDEX_TO_BASE = {
        0: 'A', 1: 'C', 2: 'G', 3: 'T'
    }
    """
    A dictionary mapping integers to members of the alphabet (i.e.
    all possible symbols that can occur in a sequence). We expect
    that `INDEX_TO_BASE[i]==BASES_ARR[i]` is `True` for all
    valid `i`.
    """

    COMPLEMENTARY_BASE_DICT = {
        'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
        'a': 'T', 'c': 'G', 'g': 'C', 't': 'A', 'n': 'N'
    }
    """
    A dictionary mapping each base to its complementary base.
    """

    UNK_BASE = "N"
    """
    This is a base used to represent unknown positions. This is not
    the same as a character from outside the sequence's alphabet. A
    character from outside the alphabet is an error. A position with
    an unknown base signifies that the position is one of the bases
    from the alphabet, but we are uncertain which.
    """

    def __init__(self, input_path, blacklist_regions=None, bases_order=None, init_unpicklable=False):
        """
        Constructs a `Genome` object.
        """

        self.input_path = input_path
        self.blacklist_regions = blacklist_regions
        self._initialized =False

        if bases_order is not None:
            bases = [str.upper(b) for b in bases_order]
            self.BASES_ARR = bases
            lc_bases = [str.lower(b) for b in bases]
            self.BASE_TO_INDEX = {
                **{b: ix for (ix, b) in enumerate(bases)},
                **{b: ix for (ix, b) in enumerate(lc_bases)}}
            self.INDEX_TO_BASE = {ix: b for (ix, b) in enumerate(bases)}
            self.update_bases_order(bases)

        if init_unpicklable:
            self._unpicklable_init()

    @classmethod
    def update_bases_order(cls, bases):
        cls.BASES_ARR = bases
        lc_bases = [str.lower(b) for b in bases]
        cls.BASE_TO_INDEX = {
            **{b: ix for (ix, b) in enumerate(bases)},
            **{b: ix for (ix, b) in enumerate(lc_bases)}}
        cls.INDEX_TO_BASE = {ix: b for (ix, b) in enumerate(bases)}

    def _unpicklable_init(self):
        if not self._initialized:
            self.genome = pyfaidx.Fasta(self.input_path)
            self.chrs = sorted(self.genome.keys())
            self.len_chrs = self._get_len_chrs()
            self._blacklist_tabix = None

            if self.blacklist_regions == "hg19":
                self._blacklist_tabix = tabix.open(
                    pkg_resources.resource_filename(
                        "selene_sdk",
                        "sequences/data/hg19_blacklist_ENCFF001TDO.bed.gz"))
            elif self.blacklist_regions == "hg38":
                self._blacklist_tabix = tabix.open(
                    pkg_resources.resource_filename(
                        "selene_sdk",
                        "sequences/data/hg38.blacklist.bed.gz"))
            elif self.blacklist_regions is not None:  # user-specified file
                self._blacklist_tabix = tabix.open(
                    self.blacklist_regions)
            self._initialized = True

    def init(func):
        # delay initialization to allow  multiprocessing
        @wraps(func)
        def dfunc(self, *args, **kwargs):
            self._unpicklable_init()
            return func(self, *args, **kwargs)
        return dfunc

    @init
    def get_chrs(self):
        """Gets the list of chromosome names.

        Returns
        -------
        list(str)
            A list of the chromosome names.

        """
        return self.chrs

    @init
    def get_chr_lens(self):
        """Gets the name and length of each chromosome sequence in the file.

        Returns
        -------
        list(tuple(str, int))
            A list of tuples of the chromosome names and lengths.

        """
        return [(k, self.len_chrs[k]) for k in self.get_chrs()]

    def _get_len_chrs(self):
        len_chrs = {}
        for chrom in self.chrs:
            len_chrs[chrom] = len(self.genome[chrom])
        return len_chrs

    def _genome_sequence(self, chrom, start, end, strand='+'):
        if strand == '+' or strand == '.':
            return self.genome[chrom][start:end].seq
        else:
            return self.genome[chrom][start:end].reverse.complement.seq

    @init
    def coords_in_bounds(self, chrom, start, end):
        """
        Check if the region we want to query is within the bounds of the
        queried chromosome and non-overlapping with blacklist regions
        (if given).

        Parameters
        ----------
        chrom : str
            The name of the chromosomes, e.g. "chr1".
        start : int
            The 0-based start coordinate of the sequence.
        end : int
            One past the 0-based last position in the sequence.

        Returns
        -------
        bool
            Whether we can retrieve a sequence from the bounds specified
            in the input.

        """
        return _check_coords(self.len_chrs,
                             chrom,
                             start,
                             end,
                             blacklist_tabix=self._blacklist_tabix)

    @init
    def get_sequence_from_coords(self,
                                 chrom,
                                 start,
                                 end,
                                 strand='+',
                                 pad=False):
        """
        Gets the queried chromosome's sequence at the input coordinates.

        Parameters
        ----------
        chrom : str
            The name of the chromosomes, e.g. "chr1".
        start : int
            The 0-based start coordinate of the sequence.
        end : int
            One past the 0-based last position in the sequence.
        strand : {'+', '-', '.'}, optional
            Default is '+'. The strand the sequence is located on. '.' is
            treated as '.'.
        pad : bool, optional
            Default is `False`. Pad the output sequence with 'N' if `start`
            and/or `end` are out of bounds to return a sequence of length
            `end - start`.

        Returns
        -------
        str
            The genomic sequence of length :math:`L` where
            :math:`L = end - start`. If `pad` is `False` and one/both of
            `start` and `end` are out of bounds, will return an empty string.
            Also returns an empty string if `chrom` cannot be found in the
            input FASTA file.
            Otherwise, will return the sequence with padding at the start/end
            if appropriate.

        Raises
        ------
        ValueError
            If the input char to `strand` is not one of the specified
            choices.

        """
        return _get_sequence_from_coords(self.len_chrs,
                                         self._genome_sequence,
                                         chrom,
                                         start,
                                         end,
                                         strand=strand,
                                         pad=pad,
                                         blacklist_tabix=self._blacklist_tabix)

    @init
    def get_encoding_from_coords(self,
                                 chrom,
                                 start,
                                 end,
                                 strand='+',
                                 pad=False):
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
        strand : {'+', '-', '.'}, optional
            Default is '+'. The strand the sequence is located on. '.' is
            treated as '+'.
        pad : bool, optional
            Default is `False`. Pad the output sequence with 'N' if `start`
            and/or `end` are out of bounds to return a sequence of length
            `end - start`.


        Returns
        -------
        numpy.ndarray, dtype=numpy.float32
            The :math:`L \\times 4` encoding of the sequence, where
            :math:`L = end - start`, unless `chrom` cannot be found
            in the input FASTA, `start` or `end` are out of bounds,
            or (if a blacklist exists) the region overlaps with a blacklist
            region. In these cases, it will return an empty encoding--that is,
            `L` = 0 for the NumPy array returned.


        Raises
        ------
        ValueError
            If the input char to `strand` is not one of the specified
            choices.
            (Raised in the call to `self.get_sequence_from_coords`)

        """
        sequence = self.get_sequence_from_coords(
            chrom, start, end, strand=strand, pad=pad)
        encoding = self.sequence_to_encoding(sequence)
        return encoding

    @init
    def get_encoding_from_coords_check_unk(self,
                                 chrom,
                                 start,
                                 end,
                                 strand='+',
                                 pad=False):
        """Gets the one-hot encoding of the genomic sequence at the
        queried coordinates and check whether the sequence contains
        unknown base(s).

        Parameters
        ----------
        chrom : str
            The name of the chromosome or region, e.g. "chr1".
        start : int
            The 0-based start coordinate of the first position in the
            sequence.
        end : int
            One past the 0-based last position in the sequence.
        strand : {'+', '-', '.'}, optional
            Default is '+'. The strand the sequence is located on. '.' is
            treated as '+'.
        pad : bool, optional
            Default is `False`. Pad the output sequence with 'N' if `start`
            and/or `end` are out of bounds to return a sequence of length
            `end - start`.


        Returns
        -------
        tuple(numpy.ndarray, bool)

            * `tuple[0]` is the :math:`L \\times 4` encoding of the sequence
            containing data of `numpy.float32` type, where
            :math:`L = end - start`, unless `chrom` cannot be found
            in the input FASTA, `start` or `end` are out of bounds,
            or (if a blacklist exists) the region overlaps with a blacklist
            region. In these cases, it will return an empty encoding--that is,
            `L` = 0 for the NumPy array returned.
            * `tuple[1]` is the boolean value that indicates whether the
            sequence contains any unknown base(s) specified in self.UNK_BASE


        Raises
        ------
        ValueError
            If the input char to `strand` is not one of the specified
            choices.
            (Raised in the call to `self.get_sequence_from_coords`)
        """
        sequence = self.get_sequence_from_coords(
            chrom, start, end, strand=strand, pad=pad)
        encoding = self.sequence_to_encoding(sequence)
        return encoding, self.UNK_BASE in sequence


    @classmethod
    def sequence_to_encoding(cls, sequence):
        """Converts an input sequence to its one-hot encoding.

        Parameters
        ----------
        sequence : str
            A nucleotide sequence of length :math:`L`

        Returns
        -------
        numpy.ndarray, dtype=numpy.float32
            The :math:`L \\times 4` one-hot encoding of the sequence.

        """
        return sequence_to_encoding(sequence, cls.BASE_TO_INDEX, cls.BASES_ARR)

    @classmethod
    def encoding_to_sequence(cls, encoding):
        """Converts an input one-hot encoding to its DNA sequence.

        Parameters
        ----------
        encoding : numpy.ndarray, dtype=numpy.float32
            An :math:`L \\times 4` one-hot encoding of the sequence,
            where :math:`L` is the length of the output sequence.

        Returns
        -------
        str
            The sequence of :math:`L` nucleotides decoded from the
            input array.

        """
        return encoding_to_sequence(encoding, cls.BASES_ARR, cls.UNK_BASE)
