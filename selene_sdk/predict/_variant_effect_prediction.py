import math

import numpy as np

from ._common import _truncate_sequence
from ._common import predict


VCF_REQUIRED_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]


# TODO: Is this a general method that might belong in utils?
def read_vcf_file(input_path,
                  strand_index=None,
                  require_strand=False,
                  output_NAs_to_file=None,
                  seq_context=None,
                  reference_sequence=None):
    """
    Read the relevant columns for a variant call format (VCF) file to
    collect variants for variant effect prediction.

    Parameters
    ----------
    input_path : str
        Path to the VCF file.
    strand_index : int or None, optional
        Default is None. By default we assume the input sequence
        surrounding a variant should be on the forward strand. If your
        model is strand-specific, you may want to specify the column number
        (0-based) in the VCF file that includes the strand corresponding
        to each variant.
    require_strand : bool, optional
        Default is False. Whether strand can be specified as '.'. If False,
        Selene accepts strand value to be '+', '-', or '.' and automatically
        treats '.' as '+'. If True, Selene skips any variant with strand '.'.
        This parameter assumes that `strand_index` has been set.
    output_NAs_to_file : str or None, optional
        Default is None. Only used if `reference_sequence` and `seq_context`
        are also not None. Specify a filepath to which invalid variants are
        written. Invalid = sequences that cannot be fetched, either because
        the exact chromosome cannot be found in the `reference_sequence` FASTA
        file or because the sequence retrieved based on the specified
        `seq_context` is out of bounds.
    seq_context : int or tuple(int, int) or None, optional
        Default is None. Only used if `reference_sequence` is not None.
        Specifies the sequence context in which the variant is centered.
        `seq_context` accepts a tuple of ints specifying the start and end
        radius surrounding the variant position or a single int if the
        start and end radius are the same length.
    reference_sequence : selene_sdk.sequences.Genome or None, optional
        Default is None. Only used if `seq_context` is not None.
        The reference genome.

    Returns
    -------
    list(tuple)
        List of variants. Tuple = (chrom, position, id, ref, alt, strand)

    """
    variants = []

    with open(input_path, 'r') as file_handle:
        lines = file_handle.readlines()
        index = 0
        for index, line in enumerate(lines):
            if '#' not in line:
                break
            if "#CHROM" in line:
                cols = line.strip().split('\t')
                if cols[:5] != VCF_REQUIRED_COLS:
                    raise ValueError(
                        "First 5 columns in file {0} were {1}. "
                        "Expected columns: {2}".format(
                            input_path, cols[:5], VCF_REQUIRED_COLS))
                index += 1
                break
        for line in lines[index:]:
            cols = line.strip().split('\t')
            if len(cols) < 5:
                continue
            chrom = str(cols[0])
            if 'CHR' == chrom[:3]:
                chrom = chrom.replace('CHR', 'chr')
            elif "chr" not in chrom:
                chrom = "chr" + chrom

            if chrom == "chrMT" and \
                    chrom not in reference_sequence.get_chrs():
                chrom = "chrM"

            pos = int(cols[1])
            name = cols[2]
            ref = cols[3]
            if ref == '-':
                ref = ""
            alt = cols[4]
            strand = '+'
            if strand_index is not None:
                if require_strand and cols[strand_index] == '.':
                    continue
                elif cols[strand_index] == '-':
                    strand = '-'

            if reference_sequence and seq_context:
                if isinstance(seq_context, int):
                    seq_context = (seq_context, seq_context)
                lhs_radius, rhs_radius = seq_context
                start = pos + len(ref) // 2 - lhs_radius
                end = pos + len(ref) // 2 + rhs_radius
                if not reference_sequence.coords_in_bounds(chrom, start, end) \
                        and output_NAs_to_file:
                    with open(output_NAs_to_file, 'a') as file_handle:
                        file_handle.write(
                            "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
                                chrom, pos, name, ref, alt, strand))
                    continue
            for a in alt.split(','):
                variants.append((chrom, pos, name, ref, a, strand))
    return variants


def _get_ref_idxs(seq_len, strand, ref_len):
    mid = None
    if strand == '-':
        mid = math.ceil(seq_len / 2)
    else:
        mid = seq_len // 2

    start_pos = mid
    if strand == '-' and ref_len > 1:
        start_pos = mid - (ref_len + 1) // 2 + 1
    elif strand == '+' and ref_len == 1:
        start_pos = mid - 1
    elif strand == '+':
        start_pos = mid - ref_len // 2 - 1
    end_pos = start_pos + ref_len
    return (start_pos, end_pos)


def _process_alt(chrom,
                 pos,
                 ref,
                 alt,
                 start,
                 end,
                 strand,
                 wt_sequence,
                 reference_sequence):
    """
    Return the encoded sequence centered at a given allele for input into
    the model.

    Parameters
    ----------
    chrom : str
        The chromosome the variant is in
    pos : int
        The position of the variant
    ref : str
        The reference allele of the variant
    alt : str
        The alternate allele
    start : int
        The start coordinate for genome query
    end : int
        The end coordinate for genome query
    strand : {'+', '-'}
        The strand the variant is on
    wt_sequence : numpy.ndarray
        The reference sequence encoding
    reference_sequence : selene_sdk.sequences.Sequence
        The reference sequence Selene queries to retrieve the model input
        sequences based on variant coordinates.

    Returns
    -------
    list(numpy.ndarray)
        A list of the encoded sequences containing alternate alleles at
        the center

    """
    if alt == '*' or alt == '-':   # indicates a deletion
        alt = ''
    ref_len = len(ref)
    alt_len = len(alt)
    if alt_len > len(wt_sequence):
        sequence = _truncate_sequence(alt, len(wt_sequence))
        return reference_sequence.sequence_to_encoding(
            sequence)

    alt_encoding = reference_sequence.sequence_to_encoding(alt)
    if ref_len == alt_len:  # substitution
        start_pos, end_pos = _get_ref_idxs(len(wt_sequence), strand, ref_len)
        sequence = np.vstack([wt_sequence[:start_pos, :],
                              alt_encoding,
                              wt_sequence[end_pos:, :]])
        return sequence
    elif alt_len > ref_len:  # insertion
        start_pos, end_pos = _get_ref_idxs(len(wt_sequence), strand, ref_len)
        sequence = np.vstack([wt_sequence[:start_pos, :],
                              alt_encoding,
                              wt_sequence[end_pos:, :]])
        trunc_s = (len(sequence) - wt_sequence.shape[0]) // 2
        trunc_e = trunc_s + wt_sequence.shape[0]
        sequence = sequence[trunc_s:trunc_e, :]
        return sequence
    else:  # deletion
        lhs = reference_sequence.get_sequence_from_coords(
            chrom,
            start - ref_len // 2 + alt_len // 2,
            pos + 1,
            strand=strand,
            pad=True)
        rhs = reference_sequence.get_sequence_from_coords(
            chrom,
            pos + 1 + ref_len,
            end + math.ceil(ref_len / 2.) - math.ceil(alt_len / 2.),
            strand=strand,
            pad=True)
        if strand == '-':
            sequence = rhs + alt + lhs
        else:
            sequence = lhs + alt + rhs
        return reference_sequence.sequence_to_encoding(
            sequence)


def _handle_standard_ref(ref_encoding,
                         seq_encoding,
                         seq_length,
                         reference_sequence,
                         strand):
    ref_len = ref_encoding.shape[0]

    start_pos, end_pos = _get_ref_idxs(seq_length, strand, ref_len)

    sequence_encoding_at_ref = seq_encoding[
        start_pos:start_pos + ref_len, :]
    references_match = np.array_equal(
        sequence_encoding_at_ref, ref_encoding)

    sequence_at_ref = None
    if not references_match:
        sequence_at_ref = reference_sequence.encoding_to_sequence(
            sequence_encoding_at_ref)
        seq_encoding[start_pos:start_pos + ref_len, :] = \
            ref_encoding
    return references_match, seq_encoding, sequence_at_ref


def _handle_long_ref(ref_encoding,
                     seq_encoding,
                     start_radius,
                     end_radius,
                     reference_sequence,
                     reverse=True):
    ref_len = ref_encoding.shape[0]
    sequence_encoding_at_ref = seq_encoding
    ref_start = ref_len // 2 - start_radius
    ref_end = ref_len // 2 + end_radius
    if not reverse:
        ref_start -= 1
        ref_end -= 1
    ref_encoding = ref_encoding[ref_start:ref_end]
    references_match = np.array_equal(
        sequence_encoding_at_ref, ref_encoding)

    sequence_at_ref = None
    if not references_match:
        sequence_at_ref = reference_sequence.encoding_to_sequence(
            sequence_encoding_at_ref)
        seq_encoding = ref_encoding
    return references_match, seq_encoding, sequence_at_ref


def _handle_ref_alt_predictions(model,
                                batch_ref_seqs,
                                batch_alt_seqs,
                                batch_ids,
                                reporters,
                                use_cuda=False):
    """
    Helper method for variant effect prediction. Gets the model
    predictions and updates the reporters.

    Parameters
    ----------
    model : torch.nn.Sequential
        The model, on mode `eval`.
    batch_ref_seqs : list(np.ndarray)
        One-hot encoded sequences with the ref base(s).
    batch_alt_seqs : list(np.ndarray)
        One-hot encoded sequences with the alt base(s).
    reporters : list(PredictionsHandler)
        List of prediction handlers.
    use_cuda : bool, optional
        Default is `False`. Specifies whether CUDA-enabled GPUs are available
        for torch to use.


    Returns
    -------
    None

    """
    batch_ref_seqs = np.array(batch_ref_seqs)
    batch_alt_seqs = np.array(batch_alt_seqs)
    ref_outputs = predict(model, batch_ref_seqs, use_cuda=use_cuda)
    alt_outputs = predict(model, batch_alt_seqs, use_cuda=use_cuda)
    for r in reporters:
        if r.needs_base_pred:
            r.handle_batch_predictions(alt_outputs, batch_ids, ref_outputs)
        else:
            r.handle_batch_predictions(alt_outputs, batch_ids)
