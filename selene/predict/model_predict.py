import itertools
import os

import numpy as np
from pyfaidx import Fasta
import torch
from torch.autograd import Variable

from .predict_handlers import DiffScoreHandler, LogitScoreHandler, \
        WritePredictionsHandler, WriteRefAltHandler
from ..sequences import Genome  # TODO: Make me generic.
from ..utils import load_features_list

ISM_COLS = ["pos", "ref", "alt"]
VCF_REQUIRED_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]
VARIANTEFFECT_COLS = ["chrom", "pos", "name", "ref", "alt"]


def in_silico_mutagenesis_sequences(sequence,
                                    mutate_n_bases=1,
                                    bases_arr=None):
    """Creates a list containing each mutation that occurs from in silico
    mutagenesis across the whole sequence.

    Please note that we have not parallelized this function yet, so runtime
    increases exponentially when you increase `mutate_n_bases`.

    Parameters
    ----------
    sequence : str
    mutate_n_bases : int, optional
        Default is 1.
    bases_arr : list or None
        List of bases. If None, uses `Genome.BASES_ARR` the DNA bases
        by default.

    Returns
    -------
    list of tuple lists
        A list of all possible mutations. Each element in the list is
        itself a list of tuples, e.g. element = [(0, 'T')] when only mutating
        1 base at a time. Each tuple is the position to mutate and the base
        with which we are replacing the reference base.

        For a sequence of length 1000, mutating 1 base at a time means that
        we return a list of length 3000.
    """
    if not bases_arr:
        bases_arr = Genome.BASES_ARR

    sequence_alts = []
    for index, ref in enumerate(sequence):
        alts = []
        for base in bases_arr:
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
    positions = []
    refs = []
    alts = []
    for (position, alt) in mutation_information:
        positions.append(str(position))
        refs.append(sequence[position])
        alts.append(alt)
    return (';'.join(positions), ';'.join(refs), ';'.join(alts))


def mutate_sequence(encoded_sequence,
                    mutation_information,
                    base_to_index=None):
    """

    Parameters
    ----------
    encoded_sequence : np.ndarray
        N-by-4 reference sequence one-hot encoding.
    mutation_information : list of tuple
        List of tuples of (int, str). Each tuple is the position to mutate and
        the base to which to mutate that position in the sequence.
    base_to_index : dict or None
        Base-to-index dictionary (str -> index). If None, uses
        `Genome.BASE_TO_INDEX` the DNA base-to-index dictionary by default.

    Returns
    -------
    np.ndarray
        N-by-4 mutated sequence one-hot encoding.
    """
    if not base_to_index:
        base_to_index = Genome.BASE_TO_INDEX
    mutated_seq = np.copy(encoded_sequence)
    for (position, alt) in mutation_information:
        replace_base = base_to_index[alt]
        mutated_seq[position, :] = 0
        mutated_seq[position, replace_base] = 1
    return mutated_seq


def reverse_strand(sequence, complementary_base_dict=None):
    """

    Parameters
    ----------
    sequence : str
    complementary_base_dict : dict or None
        Base-to-complement dictionary (str -> str). If None, uses
        `Genome.COMPLEMENTARY_BASE` the DNA dict by default.

    Returns
    -------
    str
    """
    if not complementary_base_dict:
        complementary_base_dict = Genome.COMPLEMENTARY_BASE
    reverse_bases = [complementary_base_dict for b in sequence[::-1]]
    return ''.join(reverse_bases)


def read_vcf_file(vcf_file):
    """Read the relevant columns for a VCF file to collect variants
    for variant effect prediction.

    Parameters
    ----------
    vcf_file : str
        Filepath for the VCF file

    Returns
    -------
    list of tuple
        List of variants. Tuple = (chrom, position, id, ref, alt)
    """
    variants = []

    with open(vcf_file, 'r') as file_handle:
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
                            vcf_file, cols[:5], VCF_REQUIRED_COLS))
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


def _add_sequence_surrounding_alt(alt_sequence,
                                  sequence_length,
                                  chrom,
                                  ref_start,
                                  ref_end,
                                  genome):
    alt_len = len(alt_sequence)
    add_start = int((sequence_length - alt_len) / 2)
    add_end = add_start
    if (sequence_length - alt_len) % 2 != 0:
        add_end += 1

    lhs_end = ref_start
    lhs_start = ref_start - add_start

    rhs_start = ref_end
    rhs_end = ref_end + add_end

    if not genome.sequence_in_bounds(chrom, rhs_start, rhs_end):
        # add everything to the LHS
        lhs_start = ref_start - sequence_length + alt_len
        alt_sequence = genome.get_sequence_from_coords(
            chrom, lhs_start, lhs_end) + alt_sequence
    elif not genome.sequence_in_bounds(chrom, lhs_start, lhs_end):
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
    """Score sequences and their variants using the predictions made
    by a trained model."""

    def __init__(self,
                 model,
                 sequence_length,
                 batch_size,
                 features_file,
                 trained_model_file,
                 use_cuda=False):
        self.model = model

        trained_model = torch.load(trained_model_file)
        self.model.load_state_dict(trained_model["state_dict"])
        self.model.eval()

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

        self.sequence_length = sequence_length

        self._start_radius = int(sequence_length / 2)
        self._end_radius = self._start_radius
        if sequence_length % 2 != 0:
            self._end_radius += 1

        self.batch_size = batch_size
        self.features_list = load_features_list(features_file)

    def predict(self, batch_sequences):
        """

        Parameters
        ----------
        batch_sequences : np.ndarray

        Returns
        -------
        np.ndarray
        """
        inputs = torch.Tensor(batch_sequences)
        if self.use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = self.model.forward(inputs.transpose(1, 2))
        return outputs.data.cpu().numpy()

    def _initialize_reporters(self,
                              save_data,
                              output_dir,
                              filename_prefix,
                              nonfeature_cols,
                              mode="ism"):
        reporters = []
        if "diffs" in save_data:
            filename = os.path.join(
                output_dir, f"{filename_prefix}_diffs.txt")
            diff_handler = DiffScoreHandler(
                self.features_list, nonfeature_cols, filename)
            reporters.append(diff_handler)
        if "logits" in save_data:
            filename = os.path.join(
                output_dir, f"{filename_prefix}_logits.txt")
            logit_handler = LogitScoreHandler(
                self.features_list, nonfeature_cols, filename)
            reporters.append(logit_handler)
        if "predictions" in save_data and mode == "ism":
            filename = os.path.join(
                output_dir, f"{filename_prefix}_preds.txt")
            preds_handler = WritePredictionsHandler(
                self.features_list, nonfeature_cols, filename)
            reporters.append(preds_handler)
        elif "predictions" in save_data and mode == "varianteffect":
            filename = os.path.join(
                output_dir, f"{filename_prefix}_preds")
            preds_handler = WriteRefAltHandler(
                self.features_list, nonfeature_cols, filename)
            reporters.append(preds_handler)
        return reporters

    def in_silico_mutagenesis_predict(self,
                                      sequence,
                                      base_preds,
                                      mutations_list,
                                      reporters=[]):
        """
        Parameters
        ----------
        sequence : str
        base_preds : np.ndarray
        mutations_list : list of tuple
        reporters : list of PredictionsHandler

        Returns
        -------
        None
            Writes results to files corresponding to each reporter in
            `reporters`.
        """
        current_sequence_encoding = Genome.sequence_to_encoding(sequence)
        for i in range(0, len(mutations_list), self.batch_size):
            start = i
            end = i + self.batch_size

            mutated_sequences = np.zeros(
                (self.batch_size, *current_sequence_encoding.shape))

            batch_ids = []
            for ix, mutation_info in enumerate(mutations_list[start:end]):
                mutated_seq = mutate_sequence(
                    current_sequence_encoding, mutation_info)
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
                              input_sequence,
                              save_data,
                              output_dir,
                              filename_prefix="ism",
                              mutate_n_bases=1):
        """
        Parameters
        ----------
        input_sequence : str
        save_data : list of str
        output_dir : str
        filename_prefix : str, optional
        mutate_n_bases : int, optional

        Returns
        -------
        None
        """
        n = len(input_sequence)
        if n < self.sequence_length: # Pad string length as necessary.
             diff = (self.sequence_length - n) / 2
             pad_l = int(np.floor(diff))
             pad_r = int(np.ceil(diff))
             input_sequence = ('N' * pad_l) + input_sequence + ('N' * pad_r)
        elif n > self.sequence_length:  # Extract center substring of proper length.
            start = int((n - self.sequence_length) // 2)
            end = int(start + self.sequence_length)
            input_sequence = input_sequence[start:end]

        mutated_sequences = in_silico_mutagenesis_sequences(
            input_sequence, mutate_n_bases=1)

        reporters = self._initialize_reporters(
            save_data, output_dir, filename_prefix, ISM_COLS)

        current_sequence_encoding = Genome.sequence_to_encoding(input_sequence)

        base_encoding = current_sequence_encoding.reshape(
            (1, *current_sequence_encoding.shape))
        base_preds = self.predict(base_encoding)

        predictions_reporter = reporters[-1]
        predictions_reporter.handle_batch_predictions(
            base_preds, [["NA", "NA", "NA"]])

        self.in_silico_mutagenesis_predict(
            input_sequence, base_preds, mutated_sequences, reporters=reporters)

    def in_silico_mutagenesis_from_file(self,
                                        input_path,
                                        save_data,
                                        output_dir,
                                        filename_prefix="ism",
                                        mutate_n_bases=1):
        """
        Parameters
        ----------
        input_path: str
        save_data : list of str
        output_dir : str
        filename_prefix : str, optional
        mutate_n_bases : int, optional

        Please note that we have not parallelized this function yet, so runtime
        increases exponentially when you increase `mutate_n_bases`.

        Returns
        -------
        None
        """
        fasta_file = Fasta(input_path)
        for i, fasta_record in enumerate(fasta_file):
            cur_sequence = str(fasta_record)
            n = len(cur_sequence)
            if n < self.sequence_length:
                 diff = (self.sequence_length - n) / 2
                 pad_l = int(np.floor(diff))
                 pad_r = int(np.ceil(diff))
                 cur_sequence = ('N' * pad_l) + cur_sequence + ('N' * pad_r)
            elif n > self.sequence_length:
                start = int((n - self.sequence_length) // 2)
                end = int(start + self.sequence_length)
                cur_sequence = cur_sequence[start:end]

            # Generate mut sequences and base preds.
            mutated_sequences = in_silico_mutagenesis_sequences(cur_sequence, mutate_n_bases=mutate_n_bases)
            cur_sequence_encoding = Genome.sequence_to_encoding(cur_sequence)
            base_encoding = cur_sequence_encoding.reshape(1, *cur_sequence_encoding.shape)
            base_preds = self.predict(base_encoding)

            # Write base to file, and make mut preds.
            reporters = self._initialize_reporters(save_data, output_dir,
                                                   "{0}.{1}".format(i, filename_prefix), ISM_COLS)
            predictions_reporter = reporters[-1]
            predictions_reporter.handle_batch_predictions(base_preds, [["NA", "NA", "NA"]])
            self.in_silico_mutagenesis_predict(
                cur_sequence, base_preds, mutated_sequences, reporters=reporters)
        fasta_file.close()



    def handle_ref_alt_predictions(self,
                                   batch_ref_seqs,
                                   batch_alt_seqs,
                                   batch_ids,
                                   reporters):
        """
        Parameters
        ----------
        batch_ref_seqs : list of np.ndarray
        batch_alt_seqs : list of np.ndarray
        reporters : list of PredictionsHandler

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

    def _process_alts(self, all_alts, ref, chrom, start, end,
                      reference_sequence, genome):
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
        Parameters
        ----------
        vcf_file : str
        save_data : list of str
        indexed_fasta : str
        output_dir : str or None, optional

        Returns
        -------
        None
        """
        variants = read_vcf_file(vcf_file)
        genome = Genome(indexed_fasta)

        path, filename = os.path.split(vcf_file)
        out_prefix = filename.split('.')[0]
        if not output_dir:
            output_dir = path

        reporters = self._initialize_reporters(
            save_data, output_dir, out_prefix, VARIANTEFFECT_COLS,
            mode="varianteffect")

        batch_ref_seqs = []
        batch_alt_seqs = []
        batch_ids = []
        for (chrom, pos, name, ref, alt) in variants:
            center = pos + int(len(ref) / 2)
            start = center - self._start_radius
            end = center + self._end_radius
            if not genome.sequence_in_bounds(chrom, start, end):
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
