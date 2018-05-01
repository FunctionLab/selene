import itertools
import os

import numpy as np
import torch
from torch.autograd import Variable

from .predict_handlers import DiffScoreHandler, LogitScoreHandler, \
        WritePredictionsHandler, WriteRefAltHandler
from ..sequences import Genome
from ..sequences import sequence_to_encoding


VCF_REQUIRED_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]


def in_silico_mutagenesis_sequences(sequence,
                                    mutate_n_bases=1,
                                    bases_arr=None):
    """Creates a list containing each mutation that occurs from in silico
    mutagenesis across the whole sequence.

    Parameters
    ----------
    sequence : str
    mutate_n_bases : int
    bases_arr : list|None
        List of bases (e.g. 'A', 'C', 'G', 'T' for DNA). If None, currently
        uses `Genome.BASES_ARR` the DNA bases by default.

    Returns
    -------
    list
        A list of all possible mutations. Each element in the list is
        itself a list of tuples, e.g. [(0, 'T')] if we are only mutating
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


def _ism_sample_id(dna_sequence, mutation_information):
    positions = []
    refs = []
    alts = []
    for (position, alt) in mutation_information:
        positions.append(str(position))
        refs.append(dna_sequence[position])
        alts.append(alt)
    return (';'.join(positions), ';'.join(refs), ';'.join(alts))


def mutate_sequence(encoded_sequence,
                    mutation_information,
                    base_to_index=None):
    if not base_to_index:
        base_to_index = Genome.BASE_TO_INDEX
    mutated_seq = np.copy(encoded_sequence)
    for (position, alt) in mutation_information:
        replace_base = base_to_index[alt]
        mutated_seq[position, :] = 0
        mutated_seq[position, replace_base] = 1
    return mutated_seq


def reverse_strand(sequence, complementary_base_dict=None):
    if not complementary_base_dict:
        complementary_base_dict = Genome.COMPLEMENTARY_BASE
    reverse_bases = [complementary_base_dict for b in sequence[::-1]]
    return ''.join(reverse_bases)


def read_vcf_file(vcf_file):
    """Read the relevant columns for a VCF file to collect variants
    for variant effect prediction.
    """
    variants = []

    with open(vcf_file, 'r') as file_handle:
        lines = file_handle.readlines()
        for index, line in enumerate(lines):
            if '#' not in line:
                break
            elif "#CHROM" in line:
                cols = line.strip().split('\t')
                if cols[:5] != VCF_REQUIRED_COLS:
                    raise ValueError(
                        "First 5 columns in file {0} were {1}. "
                        "Expected columns: {2}".format(
                            vcf_file, cols[:5], VCF_REQUIRED_COLS))
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


class AnalyzeSequences(object):
    """Score sequences and their variants using the predictions made
    by a trained model."""

    def __init__(self, model, batch_size, features_list, use_cuda=False):
        self.model = model
        self.batch_size = batch_size
        self.features_list = features_list
        self.use_cuda = use_cuda

    def predict(self, batch_sequences):
        inputs = torch.Tensor(batch_sequences)
        if self.use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = self.model.forward(inputs.transpose(1, 2))
        return outputs

    def _initialize_reporters(self,
                              save_diffs,
                              save_logits,
                              save_predictions,
                              output_dir,
                              filename_prefix,
                              nonfeature_cols,
                              mode="ism"):
        reporters = []
        if save_diffs:
            filename = os.path.join(
                output_dir, f"{filename_prefix}_diffs.txt")
            diff_handler = DiffScoreHandler(
                self.features_list, nonfeature_cols, filename)
            reporters.append(diff_handler)
        if save_logits:
            filename = os.path.join(
                output_dir, f"{filename_prefix}_logits.txt")
            logit_handler = LogitScoreHandler(
                self.features_list, nonfeature_cols, filename)
            reporters.append(logit_handler)
        if save_predictions and mode == "ism":
            filename = os.path.join(
                output_dir, f"{filename_prefix}_preds.txt")
            preds_handler = WritePredictionsHandler(
                self.features_list, nonfeature_cols, filename)
            reporters.append(preds_handler)
        elif save_predictions and mode == "varianteffect":
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
        current_sequence_encoding = sequence_to_encoding(
            sequence, Genome.BASE_TO_INDEX)
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
            outputs = self.predict(mutated_sequences).data.cpu().numpy()

            for r in reporters:
                if r.needs_base_pred:
                    r.handle_batch_predictions(outputs, batch_ids, base_preds)
                else:
                    r.handle_batch_predictions(outputs, batch_ids)

        for r in reporters:
            r.write_to_file()


    def in_silico_mutagenesis(self,
                              input_sequence,
                              save_diffs,
                              output_dir,
                              filename_prefix="ism",
                              mutate_n_bases=1,
                              save_logits=False,
                              save_predictions=False):
        mutated_sequences = in_silico_mutagenesis_sequences(
            input_sequence, mutate_n_bases=1)

        nonfeature_cols = ["pos", "ref", "alt"]
        reporters = self._initialize_reporters(
            save_diffs, save_logits, save_predictions,
            output_dir, filename_prefix,
            nonfeature_cols)

        current_sequence_encoding = sequence_to_encoding(
            input_sequence, Genome.BASE_TO_INDEX)

        base_encoding = current_sequence_encoding.reshape(
            (1, *current_sequence_encoding.shape))
        base_preds = self.predict(base_encoding).data.cpu().numpy()

        predictions_reporter = reporters[-1]
        predictions_reporter.handle_batch_predictions(
            base_preds, [["NA", "NA", "NA"]])

        self.in_silico_mutagenesis_predict(
            input_sequence, base_preds, mutated_sequences, reporters=reporters)

    def variant_effect_prediction(self,
                                  vcf_file,
                                  sequence_length,
                                  save_diffs,
                                  indexed_fasta,
                                  output_dir=None,
                                  save_logits=False,
                                  save_predictions=False):
        variants = read_vcf_file(vcf_file)
        genome = Genome(indexed_fasta)
        radius = int(sequence_length / 2)
        start_radius = radius
        end_radius = radius
        if sequence_length % 2 != 0:
            end_radius += 1
        path, filename = os.path.split(vcf_file)
        out_prefix = filename.split('.')[0]

        if not output_dir:
            output_dir = path

        nonfeature_cols = ["chr", "pos", "name", "ref", "alt"]
        reporters = self._initialize_reporters(
            save_diffs, save_logits, save_predictions,
            output_dir, out_prefix,
            nonfeature_cols,
            mode="varianteffect")

        batch_ref_seqs = []
        batch_alt_seqs = []
        batch_ids = []
        for (chrom, pos, name, ref, alt) in variants:
            start = pos - start_radius
            end = pos + end_radius

            if not genome.sequence_in_bounds(chrom, start, end):
                for r in reporters:
                    r.handle_NA((chrom, pos, name, ref, alt))

            reference_sequence = genome.get_sequence_from_coords(
                chrom, start, end)
            assert len(reference_sequence) == sequence_length

            ref_encoding = genome.sequence_to_encoding(reference_sequence)

            all_alts = alt.split(',')
            for a in all_alts:
                prefix = reference_sequence[:start_radius]
                suffix = reference_sequence[start_radius + len(ref):]
                alt_sequence = prefix + a + suffix


                if len(alt_sequence) > sequence_length:
                    alt_sequence = alt_sequence[:sequence_length]
                elif len(alt_sequence) < sequence_length:
                    alt_start = end
                    alt_end = end + sequence_length - len(alt_sequence)
                    add_suffix = genome.get_sequence_from_coords(
                        chrom, alt_start, alt_end)
                    alt_sequence += add_suffix
                batch_ref_seqs.append(ref_encoding)

                assert len(alt_sequence) == sequence_length
                alt_encoding = genome.sequence_to_encoding(alt_sequence)
                batch_alt_seqs.append(alt_encoding)
                batch_ids.append((chrom, pos, name, ref, a))

                if len(batch_ref_seqs) == self.batch_size:
                    batch_ref_seqs = np.array(batch_ref_seqs)
                    batch_alt_seqs = np.array(batch_alt_seqs)

                    ref_outputs = self.predict(batch_ref_seqs)
                    alt_outputs = self.predict(batch_alt_seqs)
                    for r in reporters:
                        if r.needs_base_pred:
                            r.handle_batch_predictions(
                                alt_outputs.data.cpu().numpy(), batch_ids, ref_outputs.data.cpu().numpy())
                        else:
                            r.handle_batch_predictions(alt_outputs.data.cpu().numpy(), batch_ids)

                    batch_ref_seqs = []
                    batch_alt_seqs = []
                    batch_ids = []

        batch_ref_seqs = np.array(batch_ref_seqs)
        batch_alt_seqs = np.array(batch_alt_seqs)

        ref_outputs = self.predict(batch_ref_seqs)
        alt_outputs = self.predict(batch_alt_seqs)
        for r in reporters:
            if r.needs_base_pred:
                r.handle_batch_predictions(
                    alt_outputs.data.cpu().numpy(), batch_ids, ref_outputs.data.cpu().numpy())
            else:
                r.handle_batch_predictions(alt_outputs.data.cpu().numpy(), batch_ids)

        for r in reporters:
            r.write_to_file()
