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

from .predict_handlers import AbsDiffScoreHandler
from .predict_handlers import DiffScoreHandler
from .predict_handlers import LogitScoreHandler
from .predict_handlers import WritePredictionsHandler
from .predict_handlers import WriteRefAltHandler
from ..sequences import Genome
from ..utils import load_model_from_state_dict


# TODO: MAKE THESE GENERIC:
ISM_COLS = ["pos", "ref", "alt"]
VCF_REQUIRED_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]
VARIANTEFFECT_COLS = ["chrom", "pos", "name", "ref", "alt"]


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


# TODO: Is this a general method that might belong in utils?
def read_vcf_file(input_path):
    """
    Read the relevant columns for a variant call format (VCF) file to
    collect variants for variant effect prediction.

    Parameters
    ----------
    input_path : str
        Path to the VCF file.

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
            alt = cols[4]
            variants.append((chrom, pos, name, ref, alt))
    return variants


def _add_sequence_surrounding_alt(alt_sequence,
                                  sequence_length,
                                  chrom,
                                  ref_start,
                                  ref_end,
                                  reference_sequence):
    """
    TODO

    Parameters
    ----------
    alt_sequence : str
    sequence_length : int
    chrom : str
    ref_start : int
    ref_end : int
    reference_sequence : selene_sdk.sequences.Sequence

    Returns
    -------
    str

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

    if not reference_sequence.coords_in_bounds(chrom, rhs_start, rhs_end):
        # add everything to the LHS
        lhs_start = ref_start - sequence_length + alt_len
        alt_sequence = reference_sequence.get_sequence_from_coords(
            chrom, lhs_start, lhs_end) + alt_sequence
    elif not reference_sequence.coords_in_bounds(chrom, lhs_start, lhs_end):
        # add everything to RHS
        rhs_end = ref_end + sequence_length - alt_len
        alt_sequence += reference_sequence.get_sequence_from_coords(
            chrom, rhs_start, rhs_end)
    else:
        if lhs_start >= lhs_end:
            lhs_sequence = ""
        else:
            lhs_sequence = reference_sequence.get_sequence_from_coords(
                chrom, lhs_start, lhs_end)
        rhs_sequence = reference_sequence.get_sequence_from_coords(
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
        A sequence-based model architecture.
    trained_model_path : str
        The path to the weights file for a trained sequence-based model.
        Architecture must match `model`.
    sequence_length : int
        The length of sequences that the model is expecting.
    batch_size : int, optional
        Default is 64. The size of the mini-batches to use.
    features : list(str)
        The names of the features that the model is predicting.
    use_cuda : bool, optional
        Default is `False`. Specifies whether to use CUDA or not.
    reference_sequence : class, optional
        Default is `selene_sdk.sequences.Genome`. The type of sequence that
        this analysis will be performed on. Please note that if you need
        to use variant effect prediction, you cannot only pass in the
        class--you must pass in the constructed `selene_sdk.sequences.Sequence`
        object with a particular sequence version (e.g. `Genome("hg19.fa")`).
        This version does NOT have to be the same sequence version that the
        model was trained on. That is, if the sequences in your variants file
        are hg19 but your model was trained on hg38 sequences, you should pass
        in hg19.

    Attributes
    ----------
    model : torch.nn.Module
        A sequence-based model that has already been trained.
    sequence_length : int
        The length of sequences that the model is expecting.
    batch_size : int
        The size of the mini-batches to use.
    features : list(str)
        The names of the features that the model is predicting.
    use_cuda : bool
        Specifies whether to use CUDA or not.
    reference_sequence : class
        The type of sequence on which this analysis will be performed.

    """

    def __init__(self,
                 model,
                 trained_model_path,
                 sequence_length,
                 features,
                 batch_size=64,
                 use_cuda=False,
                 reference_sequence=Genome):
        """
        Constructs a new `AnalyzeSequences` object.
        """
        trained_model = torch.load(
                trained_model_path,
                map_location=lambda storage, location: storage)

        self.model = load_model_from_state_dict(
            trained_model["state_dict"], model)
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
        self.features = features
        self.reference_sequence = reference_sequence

    def predict(self, batch_sequences):
        """
        Return model predictions for a batch of sequences.

        Parameters
        ----------
        batch_sequences : numpy.ndarray
            `batch_sequences` has the shape :math:`B \\times L \\times N`,
            where :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet.

        Returns
        -------
        numpy.ndarray
            The model predictions of shape :math:`B \\times F`, where :math:`F`
            is the number of features (classes) the model predicts.

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
        save_data : list(str)
            A list of the data files to output. Must input 1 or more of the
            following options: ["abs_diffs", "diffs", "logits", "predictions"].
        output_path_prefix : str
            TODO
        nonfeature_cols : list(str)
            TODO
        mode : str
            TODO

        Returns
        -------
        TODO
            TODO

        """
        reporters = []
        if "diffs" in save_data:
            filename = "{0}_diffs.tsv".format(output_path_prefix)
            diff_handler = DiffScoreHandler(
                self.features, nonfeature_cols, filename)
            reporters.append(diff_handler)
        if "abs_diffs" in save_data:
            filename = "{0}_abs_diffs.tsv".format(output_path_prefix)
            abs_diff_handler = AbsDiffScoreHandler(
                self.features, nonfeature_cols, filename)
            reporters.append(abs_diff_handler)
        if "logits" in save_data:
            filename = "{0}_logits.tsv".format(output_path_prefix)
            logit_handler = LogitScoreHandler(
                self.features, nonfeature_cols, filename)
            reporters.append(logit_handler)
        if "predictions" in save_data and mode != "varianteffect":
            filename = "{0}_predictions.tsv".format(output_path_prefix)
            preds_handler = WritePredictionsHandler(
                self.features, nonfeature_cols, filename)
            reporters.append(preds_handler)
        elif "predictions" in save_data and mode == "varianteffect":
            filename = "{0}_predictions".format(output_path_prefix)
            preds_handler = WriteRefAltHandler(
                self.features, nonfeature_cols, filename)
            reporters.append(preds_handler)
        return reporters

    def _pad_sequence(self, sequence):
        diff = (self.sequence_length - len(sequence)) / 2
        pad_l = int(np.floor(diff))
        pad_r = int(np.ceil(diff))
        sequence = ((self.reference_sequence.UNK_BASE * pad_l) +
                    sequence +
                    (self.reference_sequence.UNK_BASE * pad_r))
        return str.upper(sequence)

    def _truncate_sequence(self, sequence):
        start = int((len(sequence) - self.sequence_length) // 2)
        end = int(start + self.sequence_length)
        return str.upper(sequence[start:end])

    def get_predictions_for_fasta_file(self, input_path, output_dir):
        """
        Get model predictions for sequences in a FASTA file.

        Parameters
        ----------
        input_path : str
            Input path to the FASTA file.
        output_dir : str
            Output directory to write the model predictions.

        Returns
        -------
        None
            Writes the output to a file in `output_dir`.

        """
        os.makedirs(output_dir, exist_ok=True)

        _, filename = os.path.split(input_path)
        output_prefix = '.'.join(filename.split('.')[:-1])

        reporter = self._initialize_reporters(
            ["predictions"],
            os.path.join(output_dir, output_prefix),
            ["name"],
            mode="prediction")[0]
        fasta_file = pyfaidx.Fasta(input_path)
        sequences = np.zeros((self.batch_size,
                              self.sequence_length,
                              len(self.reference_sequence.BASES_ARR)))
        batch_ids = []
        for i, fasta_record in enumerate(fasta_file):
            cur_sequence = str(fasta_record)

            if len(cur_sequence) < self.sequence_length:
                cur_sequence = self._pad_sequence(cur_sequence)
            elif len(cur_sequence) > self.sequence_length:
                cur_sequence = self._truncate_sequence(cur_sequence)

            cur_sequence_encoding = self.reference_sequence.sequence_to_encoding(
                cur_sequence)
            batch_ids.append([fasta_record.name])

            if i and i % self.batch_size == 0:
                preds = self.predict(sequences)
                sequences = np.zeros(
                    self.batch_size, *cur_sequence_encoding.shape)
                reporter.handle_batch_predictions(preds, batch_ids)

            sequences[i % self.batch_size, :, :] = cur_sequence_encoding

        if i % self.batch_size != 0:
            sequences = sequences[:i % self.batch_size + 1, :, :]
            preds = self.predict(sequences)
            reporter.handle_batch_predictions(preds, batch_ids)

        fasta_file.close()
        reporter.write_to_file(close=True)


    def in_silico_mutagenesis_predict(self,
                                      sequence,
                                      base_preds,
                                      mutations_list,
                                      reporters=[]):
        """
        Get the predictions for all specified mutations applied
        to a given sequence and, if applicable, compute the scores
        ("abs_diffs", "diffs", "logits") for these mutations.

        Parameters
        ----------
        sequence : str
            The sequence to mutate.
        base_preds : numpy.ndarray
            The model's prediction for `sequence`.
        mutations_list : list(list(tuple))
            The mutations to apply to the sequence. Each element in
            `mutations_list` is a list of tuples, where each tuple
            specifies the `int` position in the sequence to mutate and what
            `str` base to which the position is mutated (e.g. (1, 'A')).
        reporters : list(PredictionsHandler)
            The list of reporters, where each reporter handles the predictions
            made for each mutated sequence. Will collect, compute scores
            (e.g. `AbsDiffScoreHandler` computes the absolute difference
            between `base_preds` and the predictions for the mutated
            sequence), and output these as a file at the end.

        Returns
        -------
        None
            Writes results to files corresponding to each reporter in
            `reporters`.

        """
        current_sequence_encoding = self.reference_sequence.sequence_to_encoding(
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
                    reference_sequence=self.reference_sequence)
                mutated_sequences[ix, :, :] = mutated_seq
                batch_ids.append(_ism_sample_id(sequence, mutation_info))
            outputs = self.predict(mutated_sequences)

            for r in reporters:
                if r.needs_base_pred:
                    r.handle_batch_predictions(outputs, batch_ids, base_preds)
                else:
                    r.handle_batch_predictions(outputs, batch_ids)

        for r in reporters:
            r.write_to_file(close=True)

    def in_silico_mutagenesis(self,
                              sequence,
                              save_data,
                              output_path_prefix="ism",
                              mutate_n_bases=1):
        """
        Applies *in silico* mutagenesis to a sequence.

        Parameters
        ----------
        sequence : str
            The sequence to mutate.
        save_data : list(str)
            A list of the data files to output. Must input 1 or more of the
            following options: ["abs_diffs", "diffs", "logits", "predictions"].
        output_path_prefix : str, optional
            The path to which the data files are written. If directories in
            the path do not yet exist they will be automatically created.
        mutate_n_bases : int, optional
            The number of bases to mutate at one time. We recommend leaving
            this parameter set to `1` at this time, as we have not yet
            optimized operations for double and triple mutations.

        Returns
        -------
        None
            Writes results to files corresponding to each reporter in
            `reporters`.

        """
        path_dirs, _ = os.path.split(output_path_prefix)
        os.makedirs(path_dirs, exists_ok=True)

        n = len(sequence)
        if n < self.sequence_length: # Pad string length as necessary.
             diff = (self.sequence_length - n) / 2
             pad_l = int(np.floor(diff))
             pad_r = int(np.ceil(diff))
             sequence = ((self.reference_sequence.UNK_BASE * pad_l) +
                         sequence +
                         (self.reference_sequence.UNK_BASE * pad_r))
        elif n > self.sequence_length:  # Extract center substring of proper length.
            start = int((n - self.sequence_length) // 2)
            end = int(start + self.sequence_length)
            sequence = sequence[start:end]

        sequence = str.upper(sequence)
        mutated_sequences = in_silico_mutagenesis_sequences(
            sequence, mutate_n_bases=1,
            reference_sequence=self.reference_sequence)

        reporters = self._initialize_reporters(
            save_data, output_path_prefix, ISM_COLS)

        current_sequence_encoding = self.reference_sequence.sequence_to_encoding(
            sequence)

        base_encoding = current_sequence_encoding.reshape(
            (1, *current_sequence_encoding.shape))
        base_preds = self.predict(base_encoding)

        if "predictions" in save_data:
            predictions_reporter = reporters[-1]
            predictions_reporter.handle_batch_predictions(
                base_preds, [["NA", "NA", "NA"]])

        self.in_silico_mutagenesis_predict(
            sequence, base_preds, mutated_sequences, reporters=reporters)

    def in_silico_mutagenesis_from_file(self,
                                        input_path,
                                        save_data,
                                        output_dir,
                                        mutate_n_bases=1,
                                        use_sequence_name=True):
        """
        Apply *in silico* mutagenesis to all sequences in a FASTA file.

        Please note that we have not parallelized this function yet, so runtime
        increases exponentially when you increase `mutate_n_bases`.

        Parameters
        ----------
        input_path: str
            The path to the FASTA file of sequences.
        save_data : list(str)
            A list of the data files to output. Must input 1 or more of the
            following options: ["abs_diffs", "diffs", "logits", "predictions"].
        output_dir : str
            The path to the output directory. Directories in the path will be
            created if they do not currently exist.
        mutate_n_bases : int, optional
            Default is 1. The number of bases to mutate at one time in
            *in silico* mutagenesis.
        use_sequence_name : bool, optional.
            Default is True. If `use_sequence_name`, output files are prefixed
            by the sequence name/description corresponding to each sequence
            in the FASTA file. Spaces in the sequence name are replaced with
            underscores '_'. If not `use_sequence_name`, output files are
            prefixed with an index :math:`i` (starting with 0) corresponding
            to the :math:`i`th sequence in the FASTA file.

        Returns
        -------
        None
            Outputs data files from *in silico* mutagenesis to `output_dir`.

        """
        os.makedirs(output_dir, exist_ok=True)

        fasta_file = pyfaidx.Fasta(input_path)
        for i, fasta_record in enumerate(fasta_file):
            cur_sequence = str.upper(str(fasta_record))
            if len(cur_sequence) < self.sequence_length:
                cur_sequence = self._pad_sequence(cur_sequence)
            elif len(cur_sequence) > self.sequence_length:
                cur_sequence = self._truncate_sequence(cur_sequence)

            # Generate mut sequences and base preds.
            mutated_sequences = in_silico_mutagenesis_sequences(
                cur_sequence, mutate_n_bases=mutate_n_bases,
                reference_sequence=self.reference_sequence)
            cur_sequence_encoding = self.reference_sequence.sequence_to_encoding(
                cur_sequence)
            base_encoding = cur_sequence_encoding.reshape(
                1, *cur_sequence_encoding.shape)
            base_preds = self.predict(base_encoding)

            file_prefix = None
            if use_sequence_name:
                file_prefix = os.path.join(
                    output_dir, fasta_record.name.replace(' ', '_'))
            else:
                file_prefix = os.path.join(
                    output_dir, str(i))
            # Write base to file, and make mut preds.
            reporters = self._initialize_reporters(
                save_data, file_prefix, ISM_COLS)

            if "predictions" in save_data:
                predictions_reporter = reporters[-1]
                predictions_reporter.handle_batch_predictions(
                    base_preds, [["NA", "NA", "NA"]])

            self.in_silico_mutagenesis_predict(
                cur_sequence, base_preds, mutated_sequences,
                reporters=reporters)
        fasta_file.close()

    def _handle_ref_alt_predictions(self,
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

    def _process_alts(self,
                      all_alts,
                      ref,
                      chrom,
                      start,
                      end):
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

        Returns
        -------
        TODO
            TODO

        """
        sequence = self.reference_sequence.get_sequence_from_coords(
            chrom, start, end)

        alt_encodings = []
        for a in all_alts:
            prefix = sequence[:self._start_radius]
            suffix = sequence[self._start_radius + len(ref):]
            if a == '*':  # indicates a deletion
                alt_sequence = prefix + suffix
            else:
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
                    chrom, start, end, self.reference_sequence)
            alt_encoding = self.reference_sequence.sequence_to_encoding(
                alt_sequence)
            alt_encodings.append(alt_encoding)
        return alt_encodings

    def variant_effect_prediction(self,
                                  vcf_file,
                                  save_data,
                                  output_dir=None):
        """
        Get model predictions and scores for a list of variants.

        Parameters
        ----------
        vcf_file : str
            Path to a VCF file. Must contain the columns
            [#CHROM, POS, ID, REF, ALT], in order. Column header does not need
            to be present.
        save_data : list(str)
            A list of the data files to output. Must input 1 or more of the
            following options: ["abs_diffs", "diffs", "logits", "predictions"].
        output_dir : str or None, optional
            Path to the output directory. If no path is specified, will save
            files corresponding to the options in `save_data` to the
            current working directory.

        Returns
        -------
        None
            Saves all files to `output_dir`.

        """
        variants = read_vcf_file(vcf_file)

        # TODO: GIVE USER MORE CONTROL OVER PREFIX.
        path, filename = os.path.split(vcf_file)
        output_path_prefix = '.'.join(filename.split('.')[:-1])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
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
            if isinstance(self.reference_sequence, Genome):
                if "chr" not in chrom:
                    chrom = "chr" + chrom
                if "MT" in chrom:
                    chrom = chrom[:-1]

            if not self.reference_sequence.coords_in_bounds(chrom, start, end):
                for r in reporters:
                    r.handle_NA((chrom, pos, name, ref, alt))
                continue
            ref_encoding = self.reference_sequence.get_encoding_from_coords(
                chrom, start, end)

            all_alts = alt.split(',')
            alt_encodings = self._process_alts(
                all_alts, ref, chrom, start, end)
            for a in all_alts:
                batch_ref_seqs.append(ref_encoding)
                batch_ids.append((chrom, pos, name, ref, a))
            batch_alt_seqs += alt_encodings

            if len(batch_ref_seqs) >= self.batch_size:
                self._handle_ref_alt_predictions(
                    batch_ref_seqs, batch_alt_seqs, batch_ids, reporters)
                batch_ref_seqs = []
                batch_alt_seqs = []
                batch_ids = []

        if batch_ref_seqs:
            self._handle_ref_alt_predictions(
                batch_ref_seqs, batch_alt_seqs, batch_ids, reporters)

        for r in reporters:
            r.write_to_file(close=True)
