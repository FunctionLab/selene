import itertools

import numpy as np

from ..sequences import Genome


def in_silico_mutagenesis_sequences(sequence,
                                    mutate_n_bases=1,
                                    reference_sequence=Genome):
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

    """
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
            range(len(sequence)), mutate_n_bases):
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
