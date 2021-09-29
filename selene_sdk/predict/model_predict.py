"""
This module provides the `AnalyzeSequences` class and supporting
methods.
"""
import math
import os
from time import time
import warnings

import numpy as np
import pyfaidx
import torch
import torch.nn as nn

from ._common import _pad_sequence
from ._common import _truncate_sequence
from ._common import get_reverse_complement
from ._common import get_reverse_complement_encoding
from ._common import predict
from ._in_silico_mutagenesis import _ism_sample_id
from ._in_silico_mutagenesis import in_silico_mutagenesis_sequences
from ._in_silico_mutagenesis import mutate_sequence
from ._variant_effect_prediction import _handle_long_ref
from ._variant_effect_prediction import _handle_standard_ref
from ._variant_effect_prediction import _handle_ref_alt_predictions
from ._variant_effect_prediction import _process_alt
from ._variant_effect_prediction import read_vcf_file
from .predict_handlers import AbsDiffScoreHandler
from .predict_handlers import DiffScoreHandler
from .predict_handlers import LogitScoreHandler
from .predict_handlers import WritePredictionsHandler
from .predict_handlers import WriteRefAltHandler
from ..sequences import Genome
from ..utils import _is_lua_trained_model
from ..utils import load_model_from_state_dict


# TODO: MAKE THESE GENERIC:
ISM_COLS = ["pos", "ref", "alt"]
VARIANTEFFECT_COLS = ["chrom", "pos", "name", "ref", "alt", "strand", "ref_match", "contains_unk"]

class AnalyzeSequences(object):
    """
    Score sequences and their variants using the predictions made
    by a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        A sequence-based model architecture.
    trained_model_path : str or list(str)
        The path(s) to the weights file for a trained sequence-based model.
        For a single path, the model architecture must match `model`. For
        a list of paths, assumes that the `model` passed in is of type
        `selene_sdk.utils.MultiModelWrapper`, which takes in a list of
        models. The paths must be ordered the same way the models
        are ordered in that list. `list(str)` input is an API-only function--
        Selene's config file CLI does not support the `MultiModelWrapper`
        functionality at this time.
    sequence_length : int
        The length of sequences that the model is expecting.
    features : list(str)
        The names of the features that the model is predicting.
    batch_size : int, optional
        Default is 64. The size of the mini-batches to use.
    use_cuda : bool, optional
        Default is `False`. Specifies whether CUDA-enabled GPUs are available
        for torch to use.
    data_parallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available for
        torch to use during training.
    reference_sequence : class, optional
        Default is `selene_sdk.sequences.Genome`. The type of sequence on
        which this analysis will be performed. Please note that if you need
        to use variant effect prediction, you cannot only pass in the
        class--you must pass in the constructed `selene_sdk.sequences.Sequence`
        object with a particular sequence version (e.g. `Genome("hg19.fa")`).
        This version does NOT have to be the same sequence version that the
        model was trained on. That is, if the sequences in your variants file
        are hg19 but your model was trained on hg38 sequences, you should pass
        in hg19.
    write_mem_limit : int, optional
        Default is 5000. Specify, in MB, the amount of memory you want to
        allocate to storing model predictions/scores. When running one of
        _in silico_ mutagenesis, variant effect prediction, or prediction,
        prediction/score handlers will accumulate data in memory and only
        write this data to files periodically. By default, Selene will write
        to files when the total amount of data (across all handlers) takes up
        5000MB of space. Please keep in mind that Selene will not monitor the
        memory needed to actually carry out the operations (e.g. variant effect
        prediction) or load the model, so `write_mem_limit` should always be
        less than the total amount of CPU memory you have available on your
        machine. For example, for variant effect prediction, we load all
        the variants in 1 file into memory before getting the predictions, so
        your machine must have enough memory to accommodate that. Another
        possible consideration is your model size and whether you are
        using it on the CPU or a CUDA-enabled GPU (i.e. setting
        `use_cuda` to True).

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
        Specifies whether to use a CUDA-enabled GPU or not.
    data_parallel : bool
        Whether to use multiple GPUs or not.
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
                 data_parallel=False,
                 reference_sequence=Genome,
                 write_mem_limit=1500):
        """
        Constructs a new `AnalyzeSequences` object.
        """
        self.model = model

        if isinstance(trained_model_path, str):
            trained_model = torch.load(
                trained_model_path,
                map_location=lambda storage, location: storage)

            load_model_from_state_dict(
                trained_model, self.model)
        elif hasattr(trained_model_path, '__len__'):
            state_dicts = []
            for mp in trained_model_path:
                state_dict = torch.load(
                    mp, map_location=lambda storage, location: storage)
                state_dicts.append(state_dict)

            for (sd, sub_model) in zip(state_dicts, self.model.sub_models):
                load_model_from_state_dict(sd, sub_model)
        else:
            raise ValueError(
                '`trained_model_path` should be a str or list of strs '
                'specifying the full paths to model weights files, but was '
                'type {0}.'.format(type(trained_model_path)))

        self.model.eval()

        self.data_parallel = data_parallel
        if self.data_parallel:
            self.model = nn.DataParallel(model)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

        self.sequence_length = sequence_length

        self._start_radius = sequence_length // 2
        self._end_radius = self._start_radius
        if sequence_length % 2 != 0:
            self._start_radius += 1

        self.batch_size = batch_size
        self.features = features
        self.reference_sequence = reference_sequence
        if type(self.reference_sequence) == Genome and \
                not self.reference_sequence._initialized:
            self.reference_sequence._unpicklable_init()
        if type(self.reference_sequence) == Genome and \
                _is_lua_trained_model(model):
            Genome.update_bases_order(['A', 'G', 'C', 'T'])
        else:  # even if not using Genome, I guess we can update?
            Genome.update_bases_order(['A', 'C', 'G', 'T'])
        self._write_mem_limit = write_mem_limit

    def _initialize_reporters(self,
                              save_data,
                              output_path_prefix,
                              output_format,
                              colnames_for_ids,
                              output_size=None,
                              mode="ism"):
        """
        Initialize the handlers to which Selene reports model predictions

        Parameters
        ----------
        save_data : list(str)
            A list of the data files to output. Must input 1 or more of the
            following options: ["abs_diffs", "diffs", "logits", "predictions"].
        output_path_prefix : str
            Path to which the reporters will output data files. Selene will
            add a prefix to the resulting filename, where the prefix is based
            on the name of the user-specified input file. This allows a user
            to distinguish between output files from different inputs when
            a user specifies the same output directory for multiple inputs.
        output_format : {'tsv', 'hdf5'}
            The desired output format. Currently Selene supports TSV and HDF5
            formats.
        colnames_for_ids : list(str)
            Specify the names of columns that will be used to identify the
            sequence for which Selene has made predictions (e.g. (chrom,
            pos, id, ref, alt) will be the column names for variant effect
            prediction outputs).
        output_size : int, optional
            The total number of rows in the output. Must be specified when
            the output_format is hdf5.
        mode : {'prediction', 'ism', 'varianteffect'}
            If saving model predictions, the handler Selene chooses for the
            task is dependent on the mode. For example, the reporter for
            variant effect prediction writes paired ref and alt predictions
            to different files.

        Returns
        -------
        list(selene_sdk.predict.predict_handlers.PredictionsHandler)
            List of reporters to update as Selene receives model predictions.

        """
        save_data = set(save_data) & set(
            ["diffs", "abs_diffs", "logits", "predictions"])
        save_data = sorted(list(save_data))
        if len(save_data) == 0:
            raise ValueError("'save_data' parameter must be a list that "
                             "contains one of ['diffs', 'abs_diffs', "
                             "'logits', 'predictions'].")
        reporters = []
        constructor_args = [self.features,
                            colnames_for_ids,
                            output_path_prefix,
                            output_format,
                            output_size,
                            self._write_mem_limit // len(save_data)]
        for i, s in enumerate(save_data):
            write_labels = False
            if i == 0:
                write_labels = True
            if "diffs" == s:
                reporters.append(DiffScoreHandler(
                    *constructor_args, write_labels=write_labels))
            elif "abs_diffs" == s:
                reporters.append(AbsDiffScoreHandler(
                    *constructor_args, write_labels=write_labels))
            elif "logits" == s:
                reporters.append(LogitScoreHandler(
                    *constructor_args, write_labels=write_labels))
            elif "predictions" == s and mode != "varianteffect":
                reporters.append(WritePredictionsHandler(
                    *constructor_args, write_labels=write_labels))
            elif "predictions" == s and mode == "varianteffect":
                reporters.append(WriteRefAltHandler(
                    *constructor_args, write_labels=write_labels))
        return reporters

    def _get_sequences_from_bed_file(self,
                                     input_path,
                                     strand_index=None,
                                     output_NAs_to_file=None,
                                     reference_sequence=None):
        """
        Get the adjusted sequence coordinates and labels corresponding
        to each row of coordinates in an input BED file. The coordinates
        specified in each row are only used to find the center position
        for the resulting sequence--all regions returned will have the
        length expected by the model.

        Parameters
        ----------
        input_path : str
            Input filepath to BED file.
        strand_index : int or None, optional
            Default is None. If sequences must be strand-specific,
            the input BED file may include a column specifying the
            strand ({'+', '-', '.'}).
        output_NAs_to_file : str or None, optional
            Default is None. Only used if `reference_sequence` is also not None.
            Specify a filepath to which invalid variants are written.
            Invalid = sequences that cannot be fetched, either because
            the exact chromosome cannot be found in the `reference_sequence` FASTA
            file or because the sequence retrieved is out of bounds or overlapping
            with any of the blacklist regions.
        reference_sequence : selene_sdk.sequences.Genome or None, optional
            Default is None. The reference genome.

        Returns
        -------
        list(tup), list(tup)
            The sequence query information (chrom, start, end, strand)
            and the labels (the index, genome coordinates, and sequence
            specified in the BED file).

        """
        sequences = []
        labels = []
        na_rows = []
        check_chr = True
        for chrom in reference_sequence.get_chrs():
            if not chrom.startswith("chr"):
                check_chr = False
                break
        with open(input_path, 'r') as read_handle:
            for i, line in enumerate(read_handle):
                cols = line.strip().split('\t')
                if len(cols) < 3:
                    na_rows.append(line)
                    continue
                chrom = cols[0]
                start = cols[1]
                end = cols[2]
                strand = '.'
                if isinstance(strand_index, int) and len(cols) > strand_index:
                    strand = cols[strand_index]
                if 'chr' not in chrom and check_chr is True:
                    chrom = "chr{0}".format(chrom)
                if not str.isdigit(start) or not str.isdigit(end) \
                        or chrom not in self.reference_sequence.genome:
                    na_rows.append(line)
                    continue
                start, end = int(start), int(end)
                mid_pos = start + ((end - start) // 2)
                seq_start = mid_pos - self._start_radius
                seq_end = mid_pos + self._end_radius
                if reference_sequence:
                    if not reference_sequence.coords_in_bounds(chrom, seq_start, seq_end):
                        na_rows.append(line)
                        continue
                sequences.append((chrom, seq_start, seq_end, strand))
                labels.append((i, chrom, start, end, strand))

        if reference_sequence and output_NAs_to_file:
            with open(output_NAs_to_file, 'w') as file_handle:
                for na_row in na_rows:
                    file_handle.write(na_row)

        return sequences, labels

    def get_predictions_for_bed_file(self,
                                     input_path,
                                     output_dir,
                                     output_format="tsv",
                                     strand_index=None):
        """
        Get model predictions for sequences specified as genome coordinates
        in a BED file. Coordinates do not need to be the same length as the
        model expected sequence input--predictions will be centered at the
        midpoint of the specified start and end coordinates.

        Parameters
        ----------
        input_path : str
            Input path to the BED file.
        output_dir : str
            Output directory to write the model predictions.
        output_format : {'tsv', 'hdf5'}, optional
            Default is 'tsv'. Choose whether to save TSV or HDF5 output files.
            TSV is easier to access (i.e. open with text editor/Excel) and
            quickly peruse, whereas HDF5 files must be accessed through
            specific packages/viewers that support this format (e.g. h5py
            Python package). Choose

                * 'tsv' if your list of sequences is relatively small
                  (:math:`10^4` or less in order of magnitude) and/or your
                  model has a small number of features (<1000).
                * 'hdf5' for anything larger and/or if you would like to
                  access the predictions/scores as a matrix that you can
                  easily filter, apply computations, or use in a subsequent
                  classifier/model. In this case, you may access the matrix
                  using `mat["data"]` after opening the HDF5 file using
                  `mat = h5py.File("<output.h5>", 'r')`. The matrix columns
                  are the features and will match the same ordering as your
                  features .txt file (same as the order your model outputs
                  its predictions) and the matrix rows are the sequences.
                  Note that the row labels (FASTA description/IDs) will be
                  output as a separate .txt file (should match the ordering
                  of the sequences in the input FASTA).

        strand_index : int or None, optional
            Default is None. If the trained model makes strand-specific
            predictions, your input file may include a column with strand
            information (strand must be one of {'+', '-', '.'}). Specify
            the index (0-based) to use it. Otherwise, by default '+' is used.


        Returns
        -------
        None
            Writes the output to file(s) in `output_dir`. Filename will
            match that specified in the filepath.

        """
        _, filename = os.path.split(input_path)
        output_prefix = '.'.join(filename.split('.')[:-1])

        seq_coords, labels = self._get_sequences_from_bed_file(
            input_path,
            strand_index=strand_index,
            output_NAs_to_file="{0}.NA".format(os.path.join(output_dir, output_prefix)),
            reference_sequence=self.reference_sequence)

        reporter = self._initialize_reporters(
            ["predictions"],
            os.path.join(output_dir, output_prefix),
            output_format,
            ["index", "chrom", "start", "end", "strand", "contains_unk"],
            output_size=len(labels),
            mode="prediction")[0]
        sequences = None
        batch_ids = []
        for i, (label, coords) in enumerate(zip(labels, seq_coords)):
            encoding, contains_unk = self.reference_sequence.get_encoding_from_coords_check_unk(
                    *coords, pad=True)
            if sequences is None:
                sequences = np.zeros((self.batch_size, *encoding.shape))
            if i and i % self.batch_size == 0:
                preds = predict(self.model, sequences, use_cuda=self.use_cuda)
                reporter.handle_batch_predictions(preds, batch_ids)
                sequences = np.zeros((self.batch_size, *encoding.shape))
                batch_ids = []
            sequences[i % self.batch_size, :, :] = encoding
            batch_ids.append(label+(contains_unk,))
            if contains_unk:
                warnings.warn(("For region {0}, "
                               "reference sequence contains unknown "
                               "base(s). --will be marked `True` in the "
                               "`contains_unk` column of the .tsv or "
                               "row_labels .txt file.").format(label))

        sequences = sequences[:i % self.batch_size + 1, :, :]
        preds = predict(self.model, sequences, use_cuda=self.use_cuda)
        reporter.handle_batch_predictions(preds, batch_ids)
        reporter.write_to_file()


    def get_predictions_for_fasta_file(self,
                                       input_path,
                                       output_dir,
                                       output_format="tsv"):
        """
        Get model predictions for sequences in a FASTA file.

        Parameters
        ----------
        input_path : str
            Input path to the FASTA file.
        output_dir : str
            Output directory to write the model predictions.
        output_format : {'tsv', 'hdf5'}, optional
            Default is 'tsv'. Choose whether to save TSV or HDF5 output files.
            TSV is easier to access (i.e. open with text editor/Excel) and
            quickly peruse, whereas HDF5 files must be accessed through
            specific packages/viewers that support this format (e.g. h5py
            Python package). Choose

                * 'tsv' if your list of sequences is relatively small
                  (:math:`10^4` or less in order of magnitude) and/or your
                  model has a small number of features (<1000).
                * 'hdf5' for anything larger and/or if you would like to
                  access the predictions/scores as a matrix that you can
                  easily filter, apply computations, or use in a subsequent
                  classifier/model. In this case, you may access the matrix
                  using `mat["data"]` after opening the HDF5 file using
                  `mat = h5py.File("<output.h5>", 'r')`. The matrix columns
                  are the features and will match the same ordering as your
                  features .txt file (same as the order your model outputs
                  its predictions) and the matrix rows are the sequences.
                  Note that the row labels (FASTA description/IDs) will be
                  output as a separate .txt file (should match the ordering
                  of the sequences in the input FASTA).

        Returns
        -------
        None
            Writes the output to file(s) in `output_dir`.

        """
        os.makedirs(output_dir, exist_ok=True)

        _, filename = os.path.split(input_path)
        output_prefix = '.'.join(filename.split('.')[:-1])

        fasta_file = pyfaidx.Fasta(input_path)
        reporter = self._initialize_reporters(
            ["predictions"],
            os.path.join(output_dir, output_prefix),
            output_format,
            ["index", "name"],
            output_size=len(fasta_file.keys()),
            mode="prediction")[0]
        sequences = np.zeros((self.batch_size,
                              self.sequence_length,
                              len(self.reference_sequence.BASES_ARR)))
        batch_ids = []
        for i, fasta_record in enumerate(fasta_file):
            cur_sequence = self._pad_or_truncate_sequence(str(fasta_record))
            cur_sequence_encoding = self.reference_sequence.sequence_to_encoding(
                cur_sequence)

            if i and i > 0 and i % self.batch_size == 0:
                preds = predict(self.model, sequences, use_cuda=self.use_cuda)
                sequences = np.zeros(
                    (self.batch_size, *cur_sequence_encoding.shape))
                reporter.handle_batch_predictions(preds, batch_ids)
                batch_ids = []

            batch_ids.append([i, fasta_record.name])
            sequences[i % self.batch_size, :, :] = cur_sequence_encoding

        sequences = sequences[:i % self.batch_size + 1, :, :]
        preds = predict(self.model, sequences, use_cuda=self.use_cuda)
        reporter.handle_batch_predictions(preds, batch_ids)

        fasta_file.close()
        reporter.write_to_file()


    def get_predictions(self,
                        input_path,
                        output_dir=None,
                        output_format="tsv",
                        strand_index=None):
        """
        Get model predictions for sequences specified as a raw sequence,
        FASTA, or BED file.

        Parameters
        ----------
        input_path : str
            A single sequence, or a path to the FASTA or BED file input.
        output_dir : str, optional
            Default is None. Output directory to write the model predictions.
            If this is left blank a raw sequence input will be assumed, though
            an output directory is required for FASTA and BED inputs.
        output_format : {'tsv', 'hdf5'}, optional
            Default is 'tsv'. Choose whether to save TSV or HDF5 output files.
            TSV is easier to access (i.e. open with text editor/Excel) and
            quickly peruse, whereas HDF5 files must be accessed through
            specific packages/viewers that support this format (e.g. h5py
            Python package). Choose

                * 'tsv' if your list of sequences is relatively small
                  (:math:`10^4` or less in order of magnitude) and/or your
                  model has a small number of features (<1000).
                * 'hdf5' for anything larger and/or if you would like to
                  access the predictions/scores as a matrix that you can
                  easily filter, apply computations, or use in a subsequent
                  classifier/model. In this case, you may access the matrix
                  using `mat["data"]` after opening the HDF5 file using
                  `mat = h5py.File("<output.h5>", 'r')`. The matrix columns
                  are the features and will match the same ordering as your
                  features .txt file (same as the order your model outputs
                  its predictions) and the matrix rows are the sequences.
                  Note that the row labels (FASTA description/IDs) will be
                  output as a separate .txt file (should match the ordering
                  of the sequences in the input FASTA).

        strand_index : int or None, optional
            Default is None. If the trained model makes strand-specific
            predictions, your input BED file may include a column with strand
            information (strand must be one of {'+', '-', '.'}). Specify
            the index (0-based) to use it. Otherwise, by default '+' is used.
            (This parameter is ignored if FASTA file is used as input.)

        Returns
        -------
        None
            Writes the output to file(s) in `output_dir`. Filename will
            match that specified in the filepath. In addition, if any base
            in the given or retrieved sequence is unknown, the row labels .txt file
            or .tsv file will mark this sequence or region as `contains_unk = True`.

        """
        if output_dir is None:
            sequence = self._pad_or_truncate_sequence(input_path)
            seq_enc = self.reference_sequence.sequence_to_encoding(sequence)
            seq_enc = np.expand_dims(seq_enc, axis=0)  # add batch size of 1
            return predict(self.model, seq_enc, use_cuda=self.use_cuda)
        elif input_path.endswith('.fa') or input_path.endswith('.fasta'):
            self.get_predictions_for_fasta_file(
                input_path, output_dir, output_format=output_format)
        else:
            self.get_predictions_for_bed_file(
                input_path,
                output_dir,
                output_format=output_format,
                strand_index=strand_index)

        return None

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
            outputs = predict(
                self.model, mutated_sequences, use_cuda=self.use_cuda)

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
                              mutate_n_bases=1,
                              output_format="tsv",
                              start_position=0,
                              end_position=None):
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
        output_format : {'tsv', 'hdf5'}, optional
            Default is 'tsv'. The desired output format.
        start_position : int, optional
            Default is 0. The starting position of the subsequence to be
            mutated.
        end_position : int or None, optional
            Default is None. The ending position of the subsequence to be
            mutated. If left as `None`, then `self.sequence_length` will be
            used.

        Returns
        -------
        None
            Outputs data files from *in silico* mutagenesis to `output_dir`.
            For HDF5 output and 'predictions' in `save_data`, an additional
            file named `*_ref_predictions.h5` will be outputted with the
            model prediction for the original input sequence.

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
            If `start_position` is not less than `self.sequence_length`.
        ValueError
            If `end_position` is greater than `self.sequence_length`.

        """
        if end_position is None:
            end_position = self.sequence_length
        if start_position >= end_position:
            raise ValueError(("Starting positions must be less than the ending "
                              "positions. Found a starting position of {0} with "
                              "an ending position of {1}.").format(start_position,
                                                                   end_position))
        if start_position < 0:
            raise ValueError("Negative starting positions are not supported.")
        if end_position < 0:
            raise ValueError("Negative ending positions are not supported.")
        if start_position >= self.sequence_length:
            raise ValueError(("Starting positions must be less than the sequence length."
                              " Found a starting position of {0} with a sequence length "
                              "of {1}.").format(start_position, self.sequence_length))
        if end_position > self.sequence_length:
            raise ValueError(("Ending positions must be less than or equal to the sequence "
                              "length. Found an ending position of {0} with a sequence "
                              "length of {1}.").format(end_position, self.sequence_length))
        if (end_position - start_position) < mutate_n_bases:
            raise ValueError(("Fewer bases exist in the substring specified by the starting "
                              "and ending positions than need to be mutated. There are only "
                              "{0} currently, but {1} bases must be mutated at a "
                              "time").format(end_position - start_position, mutate_n_bases))

        path_dirs, _ = os.path.split(output_path_prefix)
        if path_dirs:
            os.makedirs(path_dirs, exist_ok=True)

        n = len(sequence)
        if n < self.sequence_length: # Pad string length as necessary.
             diff = (self.sequence_length - n) / 2
             pad_l = int(np.floor(diff))
             pad_r = math.ceil(diff)
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
            reference_sequence=self.reference_sequence,
            start_position=start_position,
            end_position=end_position)
        reporters = self._initialize_reporters(
            save_data,
            output_path_prefix,
            output_format,
            ISM_COLS,
            output_size=len(mutated_sequences))

        current_sequence_encoding = \
            self.reference_sequence.sequence_to_encoding(sequence)

        current_sequence_encoding = current_sequence_encoding.reshape(
            (1, *current_sequence_encoding.shape))
        base_preds = predict(
            self.model, current_sequence_encoding, use_cuda=self.use_cuda)

        if "predictions" in save_data and output_format == 'hdf5':
            ref_reporter = self._initialize_reporters(
                ["predictions"],
                "{0}_ref".format(output_path_prefix),
                output_format, ["name"], output_size=1)[0]
            ref_reporter.handle_batch_predictions(
                base_preds, [["input_sequence"]])
            ref_reporter.write_to_file()
        elif "predictions" in save_data and output_format == 'tsv':
            reporters[-1].handle_batch_predictions(
                base_preds, [["input_sequence", "NA", "NA"]])

        self.in_silico_mutagenesis_predict(
            sequence,
            base_preds,
            mutated_sequences,
            reporters=reporters)

    def in_silico_mutagenesis_from_file(self,
                                        input_path,
                                        save_data,
                                        output_dir,
                                        mutate_n_bases=1,
                                        use_sequence_name=True,
                                        output_format="tsv",
                                        start_position=0,
                                        end_position=None):
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
        output_format : {'tsv', 'hdf5'}, optional
            Default is 'tsv'. The desired output format. Each sequence in
            the FASTA file will have its own set of output files, where
            the number of output files depends on the number of `save_data`
            predictions/scores specified.
        start_position : int, optional
            Default is 0. The starting position of the subsequence to be
            mutated.
        end_position : int or None, optional
            Default is None. The ending position of the subsequence to be
            mutated. If left as `None`, then `self.sequence_length` will be
            used.




        Returns
        -------
        None
            Outputs data files from *in silico* mutagenesis to `output_dir`.
            For HDF5 output and 'predictions' in `save_data`, an additional
            file named `*_ref_predictions.h5` will be outputted with the
            model prediction for the original input sequence.

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
            If `start_position` is not less than `self.sequence_length`.
        ValueError
            If `end_position` is greater than `self.sequence_length`.

        """
        if end_position is None:
            end_position = self.sequence_length
        if start_position >= end_position:
            raise ValueError(("Starting positions must be less than the ending "
                              "positions. Found a starting position of {0} with "
                              "an ending position of {1}.").format(start_position,
                                                                   end_position))
        if start_position < 0:
            raise ValueError("Negative starting positions are not supported.")
        if end_position < 0:
            raise ValueError("Negative ending positions are not supported.")
        if start_position >= self.sequence_length:
            raise ValueError(("Starting positions must be less than the sequence length."
                              " Found a starting position of {0} with a sequence length "
                              "of {1}.").format(start_position, self.sequence_length))
        if end_position > self.sequence_length:
            raise ValueError(("Ending positions must be less than or equal to the sequence "
                              "length. Found an ending position of {0} with a sequence "
                              "length of {1}.").format(end_position, self.sequence_length))
        if (end_position - start_position) < mutate_n_bases:
            raise ValueError(("Fewer bases exist in the substring specified by the starting "
                              "and ending positions than need to be mutated. There are only "
                              "{0} currently, but {1} bases must be mutated at a "
                              "time").format(end_position - start_position, mutate_n_bases))

        os.makedirs(output_dir, exist_ok=True)

        fasta_file = pyfaidx.Fasta(input_path)
        for i, fasta_record in enumerate(fasta_file):
            cur_sequence = self._pad_or_truncate_sequence(str.upper(str(fasta_record)))

            # Generate mut sequences and base preds.
            mutated_sequences = in_silico_mutagenesis_sequences(
                cur_sequence,
                mutate_n_bases=mutate_n_bases,
                reference_sequence=self.reference_sequence,
                start_position=start_position,
                end_position=end_position)
            cur_sequence_encoding = self.reference_sequence.sequence_to_encoding(
                cur_sequence)
            base_encoding = cur_sequence_encoding.reshape(
                1, *cur_sequence_encoding.shape)
            base_preds = predict(
                self.model, base_encoding, use_cuda=self.use_cuda)

            file_prefix = None
            if use_sequence_name:
                file_prefix = os.path.join(
                    output_dir, fasta_record.name.replace(' ', '_'))
            else:
                file_prefix = os.path.join(
                    output_dir, str(i))
            # Write base to file, and make mut preds.
            reporters = self._initialize_reporters(
                save_data,
                file_prefix,
                output_format,
                ISM_COLS,
                output_size=len(mutated_sequences))

            if "predictions" in save_data and output_format == 'hdf5':
                ref_reporter = self._initialize_reporters(
                    ["predictions"],
                    "{0}_ref".format(file_prefix),
                    output_format, ["name"], output_size=1)[0]
                ref_reporter.handle_batch_predictions(
                    base_preds, [["input_sequence"]])
                ref_reporter.write_to_file()
            elif "predictions" in save_data and output_format == 'tsv':
                reporters[-1].handle_batch_predictions(
                    base_preds, [["input_sequence", "NA", "NA"]])

            self.in_silico_mutagenesis_predict(
                cur_sequence, base_preds, mutated_sequences,
                reporters=reporters)
        fasta_file.close()

    def variant_effect_prediction(self,
                                  vcf_file,
                                  save_data,
                                  output_dir=None,
                                  output_format="tsv",
                                  strand_index=None,
                                  require_strand=False):
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
            Default is None. Path to the output directory. If no path is
            specified, will save files corresponding to the options in
            `save_data` to the current working directory.
        output_format : {'tsv', 'hdf5'}, optional
            Default is 'tsv'. Choose whether to save TSV or HDF5 output files.
            TSV is easier to access (i.e. open with text editor/Excel) and
            quickly peruse, whereas HDF5 files must be accessed through
            specific packages/viewers that support this format (e.g. h5py
            Python package). Choose

                * 'tsv' if your list of variants is relatively small
                  (:math:`10^4` or less in order of magnitude) and/or your
                  model has a small number of features (<1000).
                * 'hdf5' for anything larger and/or if you would like to
                  access the predictions/scores as a matrix that you can
                  easily filter, apply computations, or use in a subsequent
                  classifier/model. In this case, you may access the matrix
                  using `mat["data"]` after opening the HDF5 file using
                  `mat = h5py.File("<output.h5>", 'r')`. The matrix columns
                  are the features and will match the same ordering as your
                  features .txt file (same as the order your model outputs
                  its predictions) and the matrix rows are the sequences.
                  Note that the row labels (chrom, pos, id, ref, alt) will be
                  output as a separate .txt file.
        strand_index : int or None, optional.
            Default is None. If applicable, specify the column index (0-based)
            in the VCF file that contains strand information for each variant.
        require_strand : bool, optional.
            Default is False. Whether strand can be specified as '.'. If False,
            Selene accepts strand value to be '+', '-', or '.' and automatically
            treats '.' as '+'. If True, Selene skips any variant with strand '.'.
            This parameter assumes that `strand_index` has been set.

        Returns
        -------
        None
            Saves all files to `output_dir`. If any bases in the 'ref' column
            of the VCF do not match those at the specified position in the
            reference genome, the row labels .txt file will mark this variant
            as `ref_match = False`. If most of your variants do not match
            the reference genome, please check that the reference genome
            you specified matches the version with which the variants were
            called. The predictions can used directly if you have verified that
            the 'ref' bases specified for these variants are correct (Selene
            will have substituted these bases for those in the reference
            genome). In addition, if any base in the retrieved reference
            sequence is unknown, the row labels .txt file will mark this variant
            as `contains_unk = True`. Finally, some variants may show up in an
            'NA' file. This is because the surrounding sequence context ended up
            being out of bounds or overlapping with blacklist regions  or the
            chromosome containing the variant did not show up in the reference
            genome FASTA file.

        """
        # TODO: GIVE USER MORE CONTROL OVER PREFIX.
        path, filename = os.path.split(vcf_file)
        output_path_prefix = '.'.join(filename.split('.')[:-1])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = path

        output_path_prefix = os.path.join(output_dir, output_path_prefix)
        variants = read_vcf_file(
            vcf_file,
            strand_index=strand_index,
            require_strand=require_strand,
            output_NAs_to_file="{0}.NA".format(output_path_prefix),
            seq_context=(self._start_radius, self._end_radius),
            reference_sequence=self.reference_sequence)
        reporters = self._initialize_reporters(
            save_data,
            output_path_prefix,
            output_format,
            VARIANTEFFECT_COLS,
            output_size=len(variants),
            mode="varianteffect")

        batch_ref_seqs = []
        batch_alt_seqs = []
        batch_ids = []
        t_i = time()
        for ix, (chrom, pos, name, ref, alt, strand) in enumerate(variants):
            # centers the sequence containing the ref allele based on the size
            # of ref
            center = pos + len(ref) // 2
            start = center - self._start_radius
            end = center + self._end_radius
            ref_sequence_encoding, contains_unk = \
                self.reference_sequence.get_encoding_from_coords_check_unk(
                    chrom, start, end)

            ref_encoding = self.reference_sequence.sequence_to_encoding(ref)
            alt_sequence_encoding = _process_alt(
                chrom, pos, ref, alt, start, end,
                ref_sequence_encoding,
                self.reference_sequence)

            match = True
            seq_at_ref = None
            if len(ref) and len(ref) < self.sequence_length:
                match, ref_sequence_encoding, seq_at_ref = _handle_standard_ref(
                    ref_encoding,
                    ref_sequence_encoding,
                    self.sequence_length,
                    self.reference_sequence)
            elif len(ref) >= self.sequence_length:
                match, ref_sequence_encoding, seq_at_ref = _handle_long_ref(
                    ref_encoding,
                    ref_sequence_encoding,
                    self._start_radius,
                    self._end_radius,
                    self.reference_sequence)

            if contains_unk:
                warnings.warn("For variant ({0}, {1}, {2}, {3}, {4}, {5}), "
                           "reference sequence contains unknown base(s)"
                           "--will be marked `True` in the `contains_unk` column "
                           "of the .tsv or the row_labels .txt file.".format(
                             chrom, pos, name, ref, alt, strand))
            if not match:
                warnings.warn("For variant ({0}, {1}, {2}, {3}, {4}, {5}), "
                              "reference does not match the reference genome. "
                              "Reference genome contains {6} instead. "
                              "Predictions/scores associated with this "
                              "variant--where we use '{3}' in the input "
                              "sequence--will be marked `False` in the `ref_match` "
                              "column of the .tsv or the row_labels .txt file".format(
                                  chrom, pos, name, ref, alt, strand, seq_at_ref))
            batch_ids.append((chrom, pos, name, ref, alt, strand, match, contains_unk))
            if strand == '-':
                ref_sequence_encoding = get_reverse_complement_encoding(
                    ref_sequence_encoding,
                    self.reference_sequence.BASES_ARR,
                    self.reference_sequence.COMPLEMENTARY_BASE_DICT)
                alt_sequence_encoding = get_reverse_complement_encoding(
                    alt_sequence_encoding,
                    self.reference_sequence.BASES_ARR,
                    self.reference_sequence.COMPLEMENTARY_BASE_DICT)
            batch_ref_seqs.append(ref_sequence_encoding)
            batch_alt_seqs.append(alt_sequence_encoding)

            if len(batch_ref_seqs) >= self.batch_size:
                _handle_ref_alt_predictions(
                    self.model,
                    batch_ref_seqs,
                    batch_alt_seqs,
                    batch_ids,
                    reporters,
                    use_cuda=self.use_cuda)
                batch_ref_seqs = []
                batch_alt_seqs = []
                batch_ids = []

            if ix and ix % 10000 == 0:
                print("[STEP {0}]: {1} s to process 10000 variants.".format(
                    ix, time() - t_i))
                t_i = time()

        if batch_ref_seqs:
            _handle_ref_alt_predictions(
                self.model,
                batch_ref_seqs,
                batch_alt_seqs,
                batch_ids,
                reporters,
                use_cuda=self.use_cuda)

        for r in reporters:
            r.write_to_file()

    def _pad_or_truncate_sequence(self, sequence):
        if len(sequence) < self.sequence_length:
            sequence = _pad_sequence(
                sequence,
                self.sequence_length,
                self.reference_sequence.UNK_BASE,
            )
        elif len(sequence) > self.sequence_length:
            sequence = _truncate_sequence(sequence, self.sequence_length)

        return sequence
