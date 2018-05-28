"""
This module provides the `AnalyzeSequences` class and supporting
methods.
"""
import itertools
import os

import numpy as np
import pyfaidx
import torch
from torch.autograd import Variable

from .predict_handlers import DiffScoreHandler
from .predict_handlers import LogitScoreHandler
from .predict_handlers import WritePredictionsHandler
from .predict_handlers import WriteRefAltHandler
from ..sequences import Genome


# TODO: MAKE THESE GENERIC:
ISM_COLS = ["pos", "ref", "alt"]
VCF_REQUIRED_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]
VARIANTEFFECT_COLS = ["chrom", "pos", "name", "ref", "alt"]


def in_silico_mutagenesis_sequences(sequence,
                                    mutate_n_bases=1,
                                    sequence_type=Genome):
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
        Default is 1. TODO: ELABORATE ABOUT HOW THIS IS PAIRWISE
    sequence_type : class, optional
        Default is `selene.sequences.Genome`. The type of sequence
        that has been passed in.

    Returns
    -------
    list(list(tuple))
        A list of all possible mutations. Each element in the list is
        itself a list of tuples, e.g. element = [(0, 'T')] when only mutating
        1 base at a time. Each tuple is the position to mutate and the base
        with which we are replacing the reference base.

        For a sequence of length 1000, mutating 1 base at a time means that
        we return a list of length 3000.
    """
    sequence_alts = []
    for index, ref in enumerate(sequence):
        alts = []
        for base in sequence_type.BASES_ARR:
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


def _ism_sample_id(sequence, mutation_information):
    """
    TODO

    Parameters
    ----------
    sequence : str
        TODO
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


def mutate_sequence(encoding,
                    mutation_information,
                    sequence_type=Genome):
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
    sequence_type : class, optional
        Default is `selene.sequences.Genome`. The type of sequence that
        the input is.

    Returns
    -------
    numpy.ndarray
        An :math:`L \\times N` array holding the one-hot encoding of
        the mutated sequence.
    """
    mutated_seq = np.copy(encoding)
    for (position, alt) in mutation_information:
        replace_base = sequence_type.BASE_TO_INDEX[alt]
        mutated_seq[position, :] = 0
        mutated_seq[position, replace_base] = 1
    return mutated_seq
    # TODO: DOES THIS METHOD WORK WITH `N` where we have 0.25's?


# TODO: MAKE ME MORE GENERIC?
def read_vcf_file(input_path):
    """
    Read the relevant columns for a VCF file to collect variants
    for variant effect prediction.

    Parameters
    ----------
    input_path : str
        Path for the VCF (variant call format) file.

    Returns
    -------
    list of tuple
        List of variants. Tuple = (chrom, position, id, ref, alt)
    """
    variants = []

    with open(input_path, 'r') as file_handle:
        lines = file_handle.readlines()
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
            alt = cols[4]
            variants.append((chrom, pos, name, ref, alt))
    return variants


# TODO: MAKE GENERIC.
def _add_sequence_surrounding_alt(alt_sequence,
                                  sequence_length,
                                  chrom,
                                  ref_start,
                                  ref_end,
                                  genome):
    """
    TODO

    Parameters
    ----------
    alt_sequence
    sequence_length
    chrom
    ref_start
    ref_end
    genome

    Returns
    -------

    """
    alt_len = len(alt_sequence)
    add_start = int((sequence_length - alt_len) / 2)
    add_end = add_start
    if (sequence_length - alt_len) % 2 != 0:
        add_end += 1

    lhs_end = ref_start
    lhs_start = ref_start - add_start

    rhs_start = ref_end
    rhs_end = ref_end + add_end

    if not genome.coords_in_bounds(chrom, rhs_start, rhs_end):
        # add everything to the LHS
        lhs_start = ref_start - sequence_length + alt_len
        alt_sequence = genome.get_sequence_from_coords(
            chrom, lhs_start, lhs_end) + alt_sequence
    elif not genome.coords_in_bounds(chrom, lhs_start, lhs_end):
        # add everything to RHS
        rhs_end = ref_end + sequence_length - alt_len
        alt_sequence += genome.get_sequence_from_coords(
            chrom, rhs_start, rhs_end)
    else:
        if lhs_start >= lhs_end:
            lhs_sequence = ""
        else:
            lhs_sequence = genome.get_sequence_from_coords(
                chrom, lhs_start, lhs_end)
        rhs_sequence = genome.get_sequence_from_coords(
            chrom, rhs_start, rhs_end)
        alt_sequence = lhs_sequence + alt_sequence + rhs_sequence
    return alt_sequence


class AnalyzeSequences(object):
    """
    Score sequences and their variants using the predictions made
    by a trained model.

      Parameters
    ----------
    model : torch.nn.Module
        A sequence-based model that has already been trained.
    sequence_length : int
        The length of sequences that the model is expecting.
    batch_size : int
        The size of the mini-batches to use.
    features :
        # TODO(DOCUMENTATION): Finish.
    use_cuda : bool
        Default is `False`. Specifies whether to use CUDA or not.
    sequence_type : class, optional
        Default is `selene.sequences.Genome`. The type of sequence that
        this analysis will be performed on.


    Attributes
    ----------
    # TODO(DOCUMENATION): Finish.
    """

    def __init__(self,
                 model,
                 sequence_length,
                 batch_size,
                 features,
                 use_cuda=False,
                 sequence_type=Genome):
        """
        Constructs a new `AnalyzeSequences` object.
        """
        self.model = model
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

        self.sequence_length = sequence_length

        self._start_radius = int(sequence_length / 2)
        self._end_radius = self._start_radius
        if sequence_length % 2 != 0:
            self._end_radius += 1

        self.batch_size = batch_size
        self.features = features
        self.sequence_type = sequence_type

    def predict(self, batch_sequences):
        """# TODO(DOCUMENTATION): Finish.

        Parameters
        ----------
        batch_sequences : numpy.ndarray
            # TODO(DOCUMENTATION): Finish.

        Returns
        -------
        numpy.ndarray
            # TODO(DOCUMENTATION): Finish.
        """
        inputs = torch.Tensor(batch_sequences)
        if self.use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = self.model.forward(inputs.transpose(1, 2))
        return outputs.data.cpu().numpy()

    def _initialize_reporters(self,
                              save_data,
                              output_path_prefix,
                              nonfeature_cols,
                              mode="ism"):
        """
        TODO

        Parameters
        ----------
        save_data : TODO
            TODO
        output_path_prefix : TODO
            TODO
        nonfeature_cols : TODO
            TODO
        mode : TODO
            TODO

        Returns
        -------
        TODO
            TODO

        """
        reporters = []
        if "diffs" in save_data:
            filename = "{0}_diffs.txt".format(output_path_prefix)
            diff_handler = DiffScoreHandler(
                self.features, nonfeature_cols, filename)
            reporters.append(diff_handler)
        if "logits" in save_data:
            filename = "{0}_logits.txt".format(output_path_prefix)
            logit_handler = LogitScoreHandler(
                self.features, nonfeature_cols, filename)
            reporters.append(logit_handler)
        if "predictions" in save_data and mode == "ism":
            filename = "{0}_preds.txt".format(output_path_prefix)
            preds_handler = WritePredictionsHandler(
                self.features, nonfeature_cols, filename)
            reporters.append(preds_handler)
        elif "predictions" in save_data and mode == "varianteffect":
            filename = "{0}_preds".format(output_path_prefix)
            preds_handler = WriteRefAltHandler(
                self.features, nonfeature_cols, filename)
            reporters.append(preds_handler)
        return reporters

    def in_silico_mutagenesis_predict(self,
                                      sequence,
                                      base_preds,
                                      mutations_list,
                                      reporters=[]):
        """
        # TODO(DOCUMENTATION): Finish.

        Parameters
        ----------
        sequence : str
            TODO
        base_preds : numpy.ndarray
            TODO
        mutations_list : list(tuple)
            TODO
        reporters : list(PredictionsHandler)
            TODO

        Returns
        -------
        None
            Writes results to files corresponding to each reporter in
            `reporters`.
        """
        current_sequence_encoding = self.sequence_type.sequence_to_encoding(
            sequence)
        for i in range(0, len(mutations_list), self.batch_size):
            start = i
            end = min(i + self.batch_size, len(mutations_list))

            mutated_sequences = np.zeros(
                (end - start, *current_sequence_encoding.shape))

            batch_ids = []
            for ix, mutation_info in enumerate(mutations_list[start:end]):
                mutated_seq = mutate_sequence(
                    current_sequence_encoding, mutation_info,
                    sequence_type=self.sequence_type)
                mutated_sequences[ix, :, :] = mutated_seq
                batch_ids.append(_ism_sample_id(sequence, mutation_info))
            outputs = self.predict(mutated_sequences)

            for r in reporters:
                if r.needs_base_pred:
                    r.handle_batch_predictions(outputs, batch_ids, base_preds)
                else:
                    r.handle_batch_predictions(outputs, batch_ids)

        for r in reporters:
            r.write_to_file()

    def in_silico_mutagenesis(self,
                              sequence,
                              save_data,
                              output_path_prefix="ism",
                              mutate_n_bases=1):
        """
        # TODO(DOCUMENTATION): Finish.

        Parameters
        ----------
        sequence : str
            TODO
        save_data : list of str
            TODO
        output_path_prefix : str, optional
            TODO
        mutate_n_bases : int, optional
            TODO

        Returns
        -------
        None
            TODO
        """
        n = len(sequence)
        if n < self.sequence_length: # Pad string length as necessary.
             diff = (self.sequence_length - n) / 2
             pad_l = int(np.floor(diff))
             pad_r = int(np.ceil(diff))
             sequence = ((self.sequence_type.UNK_BASE * pad_l) +
                         sequence +
                         (self.sequence_type.UNK_BASE * pad_r))
        elif n > self.sequence_length:  # Extract center substring of proper length.
            start = int((n - self.sequence_length) // 2)
            end = int(start + self.sequence_length)
            sequence = sequence[start:end]

        mutated_sequences = in_silico_mutagenesis_sequences(
            sequence, mutate_n_bases=1,
            sequence_type=self.sequence_type)

        reporters = self._initialize_reporters(
            save_data, output_path_prefix, ISM_COLS)

        current_sequence_encoding = self.sequence_type.sequence_to_encoding(
            sequence)

        base_encoding = current_sequence_encoding.reshape(
            (1, *current_sequence_encoding.shape))
        base_preds = self.predict(base_encoding)

        predictions_reporter = reporters[-1]
        predictions_reporter.handle_batch_predictions(
            base_preds, [["NA", "NA", "NA"]])

        self.in_silico_mutagenesis_predict(
            sequence, base_preds, mutated_sequences, reporters=reporters)

    def in_silico_mutagenesis_from_file(self,
                                        input_path,
                                        save_data,
                                        output_path_prefix="ism",
                                        mutate_n_bases=1):
        """
        TODO

        Please note that we have not parallelized this function yet, so runtime
        increases exponentially when you increase `mutate_n_bases`.

        Parameters
        ----------
        input_path: str
            TODO
        save_data : list(str)
            TODO
        output_path_prefix : str, optional
            TODO
        mutate_n_bases : int, optional
            TODO

        Yields
        -------
        None
            TODO
        """
        fasta_file = pyfaidx.Fasta(input_path)
        for i, fasta_record in enumerate(fasta_file):
            cur_sequence = str(fasta_record)
            n = len(cur_sequence)
            if n < self.sequence_length:
                 diff = (self.sequence_length - n) / 2
                 pad_l = int(np.floor(diff))
                 pad_r = int(np.ceil(diff))
                 cur_sequence = ((self.sequence_type.UNK_BASE * pad_l) +
                                 cur_sequence +
                                 (self.sequence_type.UNK_BASE * pad_r))
            elif n > self.sequence_length:
                start = int((n - self.sequence_length) // 2)
                end = int(start + self.sequence_length)
                cur_sequence = cur_sequence[start:end]

            # Generate mut sequences and base preds.
            mutated_sequences = in_silico_mutagenesis_sequences(
                cur_sequence, mutate_n_bases=mutate_n_bases,
                sequence_type=self.sequence_type)
            cur_sequence_encoding = self.sequence_type.sequence_to_encoding(
                cur_sequence)
            base_encoding = cur_sequence_encoding.reshape(
                1, *cur_sequence_encoding.shape)
            base_preds = self.predict(base_encoding)

            # Write base to file, and make mut preds.
            reporters = self._initialize_reporters(
                save_data,"{0}.{1}".format(output_path_prefix, i), ISM_COLS)
            predictions_reporter = reporters[-1]
            predictions_reporter.handle_batch_predictions(
                base_preds, [["NA", "NA", "NA"]])
            self.in_silico_mutagenesis_predict(
                cur_sequence, base_preds, mutated_sequences,
                reporters=reporters)
        fasta_file.close()

    def handle_ref_alt_predictions(self,
                                   batch_ref_seqs,
                                   batch_alt_seqs,
                                   batch_ids,
                                   reporters):
        """
        TODO

        Parameters
        ----------
        batch_ref_seqs : list(np.ndarray)
            TODO
        batch_alt_seqs : list(np.ndarray)
            TODO
        reporters : list(PredictionsHandler)
            TODO

        Returns
        -------
        None
        """
        batch_ref_seqs = np.array(batch_ref_seqs)
        batch_alt_seqs = np.array(batch_alt_seqs)

        ref_outputs = self.predict(batch_ref_seqs)
        alt_outputs = self.predict(batch_alt_seqs)
        for r in reporters:
            if r.needs_base_pred:
                r.handle_batch_predictions(
                    alt_outputs, batch_ids, ref_outputs)
            else:
                r.handle_batch_predictions(alt_outputs, batch_ids)

    #TODO: Make generic.
    def _process_alts(self, all_alts, ref, chrom, start, end,
                      reference_sequence, genome):
        """
        TODO

        Parameters
        ----------
        all_alts : TODO
            TODO
        ref : TODO
            TODO
        chrom : str
            TODO
        start : TODO
            TODO
        end : TODO
            TODO
        reference_sequence : TODO
            TODO
        genome : TODO
            TODO

        Returns
        -------
        TODO
            TODO
        """
        alt_encodings = []
        for a in all_alts:
            prefix = reference_sequence[:self._start_radius]
            suffix = reference_sequence[self._start_radius + len(ref):]
            alt_sequence = prefix + a + suffix

            if len(alt_sequence) > self.sequence_length:
                # truncate on both sides equally
                midpoint = int(len(alt_sequence) / 2)
                start = midpoint - int(self.sequence_length / 2)
                end = midpoint + int(self.sequence_length / 2)
                if self.sequence_length % 2 != 0:
                    end += 1
                alt_sequence = alt_sequence[start:end]
            elif len(alt_sequence) < self.sequence_length:
                alt_sequence = _add_sequence_surrounding_alt(
                    alt_sequence, self.sequence_length,
                    chrom, start, end, genome)
            # @TODO: remove after testing
            assert len(alt_sequence) == self.sequence_length
            alt_encoding = genome.sequence_to_encoding(alt_sequence)
            alt_encodings.append(alt_encoding)
        return alt_encodings

    def variant_effect_prediction(self,
                                  vcf_file,
                                  save_data,
                                  indexed_fasta,
                                  output_dir=None):
        """
        TODO

        Parameters
        ----------
        vcf_file : str
            TODO
        save_data : list(str)
            TODO
        indexed_fasta : str
            TODO
        output_dir : str or None, optional
            TODO

        Returns
        -------
        None
        """
        variants = read_vcf_file(vcf_file)
        genome = self.sequence_type(indexed_fasta)
        # TODO: Remove this construction from here entirely.

        # TODO: GIVE USER MORE CONTROL OVER PREFIX.
        path, filename = os.path.split(vcf_file)
        output_path_prefix = filename.split('.')[0]
        if not output_dir:
            output_dir = path
        output_path_prefix = os.path.join(output_dir, output_path_prefix)
        reporters = self._initialize_reporters(
            save_data, output_path_prefix, VARIANTEFFECT_COLS,
            mode="varianteffect")

        batch_ref_seqs = []
        batch_alt_seqs = []
        batch_ids = []
        for (chrom, pos, name, ref, alt) in variants:
            center = pos + int(len(ref) / 2)
            start = center - self._start_radius
            end = center + self._end_radius
            if not genome.coords_in_bounds(chrom, start, end):
                for r in reporters:
                    r.handle_NA((chrom, pos, name, ref, alt))
                continue
            reference_sequence = genome.get_sequence_from_coords(
                chrom, start, end)

            # @TODO: remove after testing
            assert len(reference_sequence) == self.sequence_length

            ref_encoding = genome.sequence_to_encoding(reference_sequence)

            all_alts = alt.split(',')
            alt_encodings = self._process_alts(
                all_alts, ref, chrom, start, end, reference_sequence, genome)
            for a in all_alts:
                batch_ref_seqs.append(ref_encoding)
                batch_ids.append((chrom, pos, name, ref, a))
            batch_alt_seqs += alt_encodings

            if len(batch_ref_seqs) >= self.batch_size:
                self.handle_ref_alt_predictions(
                    batch_ref_seqs, batch_alt_seqs, batch_ids, reporters)
                batch_ref_seqs = []
                batch_alt_seqs = []
                batch_ids = []

        if batch_ref_seqs:
            self.handle_ref_alt_predictions(
                batch_ref_seqs, batch_alt_seqs, batch_ids, reporters)

        for r in reporters:
            r.write_to_file()
