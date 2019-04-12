import math

import numpy as np

from ._common import _truncate_sequence
from ._common import predict


VCF_REQUIRED_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]


# TODO: Is this a general method that might belong in utils?
def read_vcf_file(input_path, strand_index=None):
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

    Returns
    -------
    list(tuple)
        List of variants. Tuple = (chrom, position, id, ref, alt)

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
            pos = int(cols[1])
            name = cols[2]
            ref = cols[3]
            if ref == '-':
                ref = ""
            alt = cols[4]
            strand = '.'
            if strand_index is not None and cols[strand_index] == '-':
                strand = '-'
            elif strand_index is not None and cols[strand_index] == '+':
                strand = '+'
            elif strand_index is not None:
                continue
            variants.append((chrom, pos, name, ref, alt, strand))
    return variants


def _process_alts(all_alts,
                  ref,
                  chrom,
                  pos,
                  ref_seq_center,
                  strand,
                  start_radius,
                  end_radius,
                  reference_sequence):
    """
    Iterate through the alternate alleles of the variant and return
    the encoded sequences centered at those alleles for input into
    the model.

    Parameters
    ----------
    all_alts : list(str)
        The list of alternate alleles corresponding to the variant
    ref : str
        The reference allele of the variant
    chrom : str
        The chromosome the variant is in
    pos : int
        The position of the variant
    ref_seq_center : int
        The center position of the sequence containing the reference allele
    strand : {'+', '-'}
        The strand the variant is on
    start_radius : int
        The number of bases to query on the LHS of the variant.
    end_radius : int
        The number of bases to query on the RHS of the variant.
    reference_sequence : selene_sdk.sequences.Sequence
        The reference sequence Selene queries to retrieve the model input
        sequences based on variant coordinates.

    Returns
    -------
    list(numpy.ndarray)
        A list of the encoded sequences containing alternate alleles at
        the center

    """
    alt_encodings = []
    for a in all_alts:
        if a == '*' or a == '-':   # indicates a deletion
            a = ''
        ref_len = len(ref)
        alt_len = len(a)
        sequence = None
        if alt_len > start_radius + end_radius:
            sequence = _truncate_sequence(a, start_radius + end_radius)
        elif ref_len == alt_len:  # substitution
            start_pos = ref_seq_center - start_radius
            end_pos = ref_seq_center + end_radius
            sequence = reference_sequence.get_sequence_from_coords(
                chrom, start_pos, end_pos, strand=strand)
            remove_ref_start = start_radius - ref_len // 2 - 1
            sequence = (sequence[:remove_ref_start] +
                        a +
                        sequence[remove_ref_start + ref_len:])
        else:  # insertion or deletion
            seq_lhs = reference_sequence.get_sequence_from_coords(
                chrom,
                pos - 1 - start_radius + alt_len // 2,
                pos - 1,
                strand=strand,
                pad=True)
            seq_rhs = reference_sequence.get_sequence_from_coords(
                chrom,
                pos - 1 + len(ref),
                pos - 1 + len(ref) + end_radius - math.ceil(alt_len / 2.),
                strand=strand,
                pad=True)
            sequence = seq_lhs + a + seq_rhs
        alt_encoding = reference_sequence.sequence_to_encoding(
            sequence)
        alt_encodings.append(alt_encoding)
    return alt_encodings


def _handle_standard_ref(ref_encoding,
                         seq_encoding,
                         start_radius,
                         reference_sequence,
                         reverse=True):
    ref_len = ref_encoding.shape[0]
    start_pos = start_radius - ref_len // 2
    if not reverse:
        start_pos -= 1
    sequence_encoding_at_ref = seq_encoding[
        start_pos:start_pos + ref_len, :]
    sequence_at_ref = reference_sequence.encoding_to_sequence(
        sequence_encoding_at_ref)
    references_match = np.array_equal(
        sequence_encoding_at_ref, ref_encoding)
    if not references_match:
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
    sequence_at_ref = reference_sequence.encoding_to_sequence(
        sequence_encoding_at_ref)
    ref_start = ref_len // 2 - start_radius
    ref_end = ref_len // 2 + end_radius
    if not reverse:
        ref_start -= 1
        ref_end -= 1
    ref_encoding = ref_encoding[ref_start:ref_end]
    references_match = np.array_equal(
        sequence_encoding_at_ref, ref_encoding)
    if not references_match:
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
    warn : bool, optional
        Whether a warning was raised or not. If `warn`, directs handlers
        to divert the predictions/scores to different files
        (filename prefixed by 'warning.') so that users
        know that Selene detected an issue with these variants.
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
