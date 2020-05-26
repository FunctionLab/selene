import itertools

import numpy as np

from ..sequences import Genome


def in_silico_mutagenesis_sequences(sequence,
                                    mutate_n_bases=1,
                                    reference_sequence=Genome,
                                    start_position=0,
                                    end_position=None):
    """
    Creates a list containing each mutation that occurs from an
    *in silico* mutagenesis across the whole sequence.

    Please note that we have not parallelized this function yet, so
    runtime increases exponentially when you increase `mutate_n_bases`.

    Parameters
    ----------
    sequence : str
        A string containing the sequence we would like to mutate.
    mutate_n_bases : int, optional
        Default is 1. The number of base changes to make with each set of
        mutations evaluated, e.g. `mutate_n_bases = 2` considers all
        pairs of SNPs.
    reference_sequence : class, optional
        Default is `selene_sdk.sequences.Genome`. The type of sequence
        that has been passed in.
    start_position : int, optional
        Default is 0. The starting position of the subsequence to be
        mutated.
    end_position : int or None, optional
        Default is None. The ending position of the subsequence to be
        mutated. If left as `None`, then `len(sequence)` will be
        used.

    Returns
    -------
    list(list(tuple))
        A list of all possible mutations. Each element in the list is
        itself a list of tuples, e.g. element = [(0, 'T')] when only mutating
        1 base at a time. Each tuple is the position to mutate and the base
        with which we are replacing the reference base.

        For a sequence of length 1000, mutating 1 base at a time means that
        we return a list with length of 3000-4000, depending on the number of
        unknown bases in the input sequences.

    Raises
    ------
    ValueError
        If the value of `start_position` or `end_position` is negative.
    ValueError
        If there are fewer than `mutate_n_bases` between `start_position`
        and `end_position`.
    ValueError
        If `start_position` is greater or equal to `end_position`.
    ValueError
        If `start_position` is not less than `len(sequence)`.
    ValueError
        If `end_position` is greater than `len(sequence)`.

    """
    if end_position is None:
        end_position = len(sequence)
    if start_position >= end_position:
        raise ValueError(("Starting positions must be less than the ending "
                          "positions. Found a starting position of {0} with "
                          "an ending position of {1}.").format(start_position,
                                                               end_position))
    if start_position < 0:
        raise ValueError("Negative starting positions are not supported.")
    if end_position < 0:
        raise ValueError("Negative ending positions are not supported.")
    if start_position >= len(sequence):
        raise ValueError(("Starting positions must be less than the sequence length."
                          " Found a starting position of {0} with a sequence length "
                          "of {1}.").format(start_position, len(sequence)))
    if end_position > len(sequence):
        raise ValueError(("Ending positions must be less than or equal to the sequence "
                          "length. Found an ending position of {0} with a sequence "
                          "length of {1}.").format(end_position, len(sequence)))
    if (end_position - start_position) < mutate_n_bases:
        raise ValueError(("Fewer bases exist in the substring specified by the starting "
                          "and ending positions than need to be mutated. There are only "
                          "{0} currently, but {1} bases must be mutated at a "
                          "time").format(end_position - start_position, mutate_n_bases))

    sequence_alts = []
    for index, ref in enumerate(sequence):
        alts = []
        for base in reference_sequence.BASES_ARR:
            if base == ref:
                continue
            alts.append(base)
        sequence_alts.append(alts)
    all_mutated_sequences = []
    for indices in itertools.combinations(
            range(start_position, end_position), mutate_n_bases):
        pos_mutations = []
        for i in indices:
            pos_mutations.append(sequence_alts[i])
        for mutations in itertools.product(*pos_mutations):
            all_mutated_sequences.append(list(zip(indices, mutations)))
    return all_mutated_sequences


def mutate_sequence(encoding,
                    mutation_information,
                    reference_sequence=Genome):
    """
    Transforms a sequence with a set of mutations.

    Parameters
    ----------
    encoding : numpy.ndarray
        An :math:`L \\times N` array (where :math:`L` is the sequence's
        length and :math:`N` is the size of the sequence type's
        alphabet) holding the one-hot encoding of the
        reference sequence.
    mutation_information : list(tuple)
        List of tuples of (`int`, `str`). Each tuple is the position to
        mutate and the base to which to mutate that position in the
        sequence.
    reference_sequence : class, optional
        Default is `selene_sdk.sequences.Genome`. A reference sequence
        from which to retrieve smaller sequences..

    Returns
    -------
    numpy.ndarray
        An :math:`L \\times N` array holding the one-hot encoding of
        the mutated sequence.

    """
    mutated_seq = np.copy(encoding)
    for (position, alt) in mutation_information:
        replace_base = reference_sequence.BASE_TO_INDEX[alt]
        mutated_seq[position, :] = 0
        mutated_seq[position, replace_base] = 1
    return mutated_seq


def _ism_sample_id(sequence, mutation_information):
    """
    TODO

    Parameters
    ----------
    sequence : str
        The input sequence to mutate.
    mutation_information : list(tuple)
        TODO

    Returns
    -------
    TODO
        TODO

    """
    positions = []
    refs = []
    alts = []
    for (position, alt) in mutation_information:
        positions.append(str(position))
        refs.append(sequence[position])
        alts.append(alt)
    return (';'.join(positions), ';'.join(refs), ';'.join(alts))
