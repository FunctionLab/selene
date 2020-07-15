"""
This module provides the RandomPositionsSampler class.

TODO: Currently, only works with sequences from `selene_sdk.sequences.Genome`.
We would like to generalize this to `selene_sdk.sequences.Sequence` if possible.
"""
from collections import namedtuple
from collections import defaultdict
import logging
import random

import numpy as np

from .online_sampler import OnlineSampler
from ..utils import get_indices_and_probabilities

logger = logging.getLogger(__name__)


SampleIndices = namedtuple(
    "SampleIndices", ["indices", "weights"])
"""
A tuple containing the indices for some samples, and a weight to
allot to each index when randomly drawing from them.

TODO: this is common to both the intervals sampler and the
random positions sampler. Can we move this to utils or
somewhere else?

Parameters
----------
indices : list(int)
    The numeric index of each sample.
weights : list(float)
    The amount of weight assigned to each sample.

Attributes
----------
indices : list(int)
    The numeric index of each sample.
weights : list(float)
    The amount of weight assigned to each sample.

"""

class RandomPositionsSampler(OnlineSampler):
    """This sampler randomly selects a position in the genome and queries for
    a sequence centered at that position for input to the model.

    TODO: generalize to selene_sdk.sequences.Sequence?

    Parameters
    ----------
    reference_sequence : selene_sdk.sequences.Genome
        A reference sequence from which to create examples.
    target_path : str
        Path to tabix-indexed, compressed BED file (`*.bed.gz`) of genomic
        coordinates mapped to the genomic features we want to predict.
    features : list(str)
        List of distinct features that we aim to predict.
    train_size : int
        Total sample size for train set.
    validation_size : int
        Total sample size of validation set.
    test_size : int
        Total sample size of test set.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    validation_holdout : list(str) or float, optional
        Default is `['chr6', 'chr7']`. Holdout can be regional or
        proportional. If regional, expects a list (e.g. `['chrX', 'chrY']`).
        Regions must match those specified in the first column of the
        tabix-indexed BED file. If proportional, specify a percentage
        between (0.0, 1.0). Typically 0.10 or 0.20.
    test_holdout : list(str) or float, optional
        Default is `['chr8', 'chr9']`. See documentation for
        `validation_holdout` for additional information.
    sequence_length : int, optional
        Default is 1000. Model is trained on sequences of `sequence_length`
        where genomic features are annotated to the center regions of
        these sequences.
    center_bin_to_predict : int, optional
        Default is 200. Query the tabix-indexed file for a region of
        length `center_bin_to_predict`.
    feature_thresholds : float [0.0, 1.0], optional
        Default is 0.5. The `feature_threshold` to pass to the
        `GenomicFeatures` object.
    mode : {'train', 'validate', 'test'}
        Default is `'train'`. The mode to run the sampler in.
    save_datasets : list(str), optional
        Default is `['test']`. The list of modes for which we should
        save the sampled data to file.
    output_dir : str or None, optional
        Default is None. The path to the directory where we should
        save sampled examples for a mode. If `save_datasets` is
        a non-empty list, `output_dir` must be specified. If
        the path in `output_dir` does not exist it will be created
        automatically.

    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Genome
        The reference sequence that examples are created from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features that we
        would like to predict.
    validation_holdout : list(str) or float
        The samples to hold out for validating model performance. These
        can be "regional" or "proportional". If regional, this is a list
        of region names (e.g. `['chrX', 'chrY']`). These regions must
        match those specified in the first column of the tabix-indexed
        BED file. If proportional, this is the fraction of total samples
        that will be held out.
    test_holdout : list(str) or float
        The samples to hold out for testing model performance. See the
        documentation for `validation_holdout` for more details.
    sequence_length : int
        The length of the sequences to  train the model on.
    bin_radius : int
        From the center of the sequence, the radius in which to detect
        a feature annotation in order to include it as a sample's label.
    surrounding_sequence_radius : int
        The length of sequence falling outside of the feature detection
        bin (i.e. `bin_radius`) center, but still within the
        `sequence_length`.
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.
    """
    def __init__(self,
                 reference_sequence,
                 target_path,
                 features,
                 train_size,
                 validation_size,
                 test_size,
                 seed=436,
                 validation_holdout=['chr6', 'chr7'],
                 test_holdout=['chr8', 'chr9'],
                 sequence_length=1000,
                 center_bin_to_predict=200,
                 feature_thresholds=0.5,
                 mode="train",
                 save_datasets=[],
                 output_dir=None):
        super(RandomPositionsSampler, self).__init__(
            reference_sequence,
            target_path,
            features,
            seed=seed,
            validation_holdout=validation_holdout,
            test_holdout=test_holdout,
            sequence_length=sequence_length,
            center_bin_to_predict=center_bin_to_predict,
            feature_thresholds=feature_thresholds,
            mode=mode,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self.sample_from_intervals = []
        self.interval_lengths = []

        self._num_chroms = 0
        self._genome_n_bases = 0
        for chrom, len_chrom in self.reference_sequence.get_chr_lens():
            self._num_chroms += 1
            self._genome_n_bases += len_chrom

        self._validation_holdout = validation_holdout
        self._N_validation = validation_size

        if test_holdout:
            self._test_holdout = test_holdout
            self._N_test = test_size

        self._N_train = train_size

        self._partition_ixs = {"Train": np.zeros(self._N_train, dtype=np.int64),
                                "Validate": np.zeros(self._N_validation, dtype=np.int64),
                                "Test": np.zeros(self._N_test, dtype=np.int64)}

        # Information about each chromosome, "Total" holds the chrom names
        # "Starts" holds the start indices in a theoretical flat array of all
        # chromosomes, "Ends" holds the end indices.
        self.chroms_info = {"Total": [],
                            "Starts": np.zeros(self._num_chroms),
                            "Ends": np.zeros(self._num_chroms)}

        if isinstance(validation_holdout, float):
            self._partition_by_proportion()
        elif isinstance(validation_holdout):
            self._partition_by_chromosome()

    # Setup `self.chroms`, `self.chroms_starts`, `self.chroms_ends`, `self.genome_length`
    def _init_chroms(self):
        tot_len = 0
        counter = 0
        for chrom, len_chrom in self.reference_sequence.get_chr_lens():
            self.chroms_info["Total"].append(chrom)
            self.chroms_info["Starts"][counter] = tot_len
            self.chroms_info["Ends"][counter] = tot_len + len_chrom
            tot_len += len_chrom
            counter += 1

    # Compute the number of elements in the genome array that belong to
    # each mode, by proportion

    # Edit to be used by chromosome code as well.
    def _assign_proportions(self):
        if self.test_holdout:
            test_prop_N = self._genome_n_bases * self._test_holdout
            validation_prop_N = self._genome_n_bases * self._validation_holdout
            training_prop_N = \
                self._genome_n_bases * (1 - self._test_holdout - self._validation_holdout)
            return int(test_prop_N), int(validation_prop_N), int(training_prop_N)
        else:
            validation_prop_N = self._genome_n_bases * self._validation_holdout
            training_prop_N = self._genome_n_bases * (1 - _validation_holdout)
            return int(validation_prop_N), int(training_prop_N)

    # def _assign_samples_per_mode(self, prop_N, mode, start, sample_size, genome_positions_arr):
    #     get_N = min(prop_N, sample_size)
    #     self._partition_ixs[mode] = genome_positions_arr[start:start + get_N]
    #     return get_N


    def _partition_by_proportion(self):
        self._init_chroms()
        self._assign_samples() # can these be expanded to be used by partition by chrom?

    def _psuedoshuffle(self):
        total = self._genome_n_bases
        test = np.arange(total, dtype=np.int64)
        jump = 250000000
        num_steps = int(total / jump)
        start = 0
        for i in range(num_steps):
            np.random.shuffle(test[start : start + jump])
            start = start + jump

        start = int(jump / 2)
        for i in range(num_steps - 1):
            np.random.shuffle(test[start : start + jump])
            start = start + jump

        return test
        # rename

    # Order: test, validate, train
    # Make sure this matches chromosome implementation
    def _assign_samples(self):
        genome_positions_arr = self._psuedoshuffle()
        start = 0
        if self.test_holdout:
            test_prop_N, validation_prop_N, training_prop_N = self._assign_proportions()
            N_test = self._N_test
            while N_test:
                get_N = min(test_prop_N, N_test)
                self._partition_ixs["test"] = genome_positions_arr[start : start + get_N]
                # get_N = self._assign_samples_per_mode(test_prop_N, "test", start, self._N_test, genome_positions_arr)
                N_test -= get_N
                if N_test :
                    start = start + get_N

            start = start + get_N

        else:
            validation_prop_N, training_prop_N = _assign_proportions()

        N_validation = self._N_validation
        while N_validation:
            get_N = min(validation_prop_N, N_validation)
            self._partition_ixs["validate"] = genome_positions_arr[start:start + get_N]
            # get_N = _assign_samples_per_mode(validation_prop_N, "validation", start, self._N_validation, genome_positions_arr)
            N_validation -= get_N # I dont think we can alter an int like this in a function bc its pass by copy
            if N_validation:
                start = start + get_N


        start = start + get_N
        while self._N_train:
            get_N = min(training_prop_N, self._N_train)
            self._partition_ixs["train"] = genome_positions_arr[start:start + get_N]
            # get_N = _assign_samples_per_mode(training_prop_N, "train", start, self._N_train, genome_positions_arr)
            self._N_train -= get_N


    # def _partition_by_chrom(self):
    #     return
    #
    #     genome_arr = np.arange(self._genome_n_bases)
    #     # to keep this SIMPLE, you create the chromosome to position map based on what chromosomes
    #     # the user specifies as holdouts!
    #     # so validation is always after training, test chroms always after validation
    #     tot_len = 0
    #     train_counter = 0
    #     validation_counter = self._N_train
    #     test_counter = self._N_train + self._N_validation
    #     genome_positions_arr = np.zeros(self._genome_n_bases)
    #
    #     for chrom, len in self.genome.get_chr_lens():
    #
    #         if chrom in self.validation_holdout:
    #             genome_positions_arr[validation_counter  : validation_counter  + len] = np.arange(tot_len, tot_len + len)
    #         elif chrom in self.test_holdout:
    #             genome_positions_arr[test_counter : test_counter + len] = np.arange(tot_len, tot_len + len)
    #         else:
    #             genome_positions_arr[train_counter  : train_counter  + len] = np.arange(tot_len, tot_len + len)
    #
    #         for mode in self.modes:
    #             # Shuffle each partitition individually
    #             np.shuffle(genome_positions_arr[0:self._N_train])
    #             np.shuffle(genome_positions_arr[self._N_train:self._N_train + self._N_validation])
    #             np.shuffle(genome_positions_arr[self._N_train + self._N_validation:])
    #
    #             # what would be interval lengths here???
    #             sample_indices = self._partition_ixs[mode].indices
    #             indices, weights = get_indices_and_probabilities(
    #                 self.interval_lengths, sample_indices)
    #             self._sample_from_mode[mode] = \
    #                 self._sample_from_mode[mode]._replace(
    #                     indices=indices, weights=weights)

    def _retrieve(self, chrom, position):
        bin_start = position - self._start_radius
        bin_end = position + self._end_radius
        retrieved_targets = self.target.get_feature_data(
            chrom, bin_start, bin_end)
        window_start = bin_start - self.surrounding_sequence_radius
        window_end = bin_end + self.surrounding_sequence_radius
        if window_end - window_start < self.sequence_length:
            print(bin_start, bin_end,
                  self._start_radius, self._end_radius,
                  self.surrounding_sequence_radius)
            return None
        strand = self.STRAND_SIDES[random.randint(0, 1)]
        retrieved_seq = \
            self.reference_sequence.get_encoding_from_coords(
                chrom, window_start, window_end, strand)
        if retrieved_seq.shape[0] == 0:
            logger.info("Full sequence centered at {0} position {1} "
                        "could not be retrieved. Sampling again.".format(
                            chrom, position))
            return None
        elif np.sum(retrieved_seq) / float(retrieved_seq.shape[0]) < 0.60:
            logger.info("Over 30% of the bases in the sequence centered "
                        "at {0} position {1} are ambiguous ('N'). "
                        "Sampling again.".format(chrom, position))
            return None

        if retrieved_seq.shape[0] < self.sequence_length:
            # TODO: remove after investigating this bug.
            print("Warning: sequence retrieved for {0}, {1}, {2}, {3} "
                  "had length less than required sequence length {4}. "
                  "This bug will be investigated and addressed in the next "
                  "version of Selene.".format(
                      chrom, window_start, window_end, strand,
                      self.sequence_length))
            return None

        if self.mode in self._save_datasets:
            feature_indices = ';'.join(
                [str(f) for f in np.nonzero(retrieved_targets)[0]])
            self._save_datasets[self.mode].append(
                [chrom,
                 window_start,
                 window_end,
                 strand,
                 feature_indices])
            if len(self._save_datasets[self.mode]) > 200000:
                self.save_dataset_to_file(self.mode)
        return (retrieved_seq, retrieved_targets)

    # Returns (chrom, poition) pair corresponding to index in the genome array.
    def _pair_from_index(self, index):
        curr_index = 0
        all = np.where(self.chroms_info["Ends"] >= index)
        min = all[0][0]
        chrom = self.chroms_info["Total"][min]
        pos = int(index - self.chroms_info["Starts"][min])
        return chrom, pos

    def _sample(self):
        # @ Kathy, where do we set the partition for a particular sample?
        # random_index = np.random.randint(0, len(self._partition_ixs[self.mode]))
        sample_index = self._partition_ixs[self.mode][0]
        chrom, pos = self._pair_from_index(sample_index)
        np.delete(self._partition_ixs[self.mode], 0)
        return chrom, pos

    def sample(self, batch_size):
        """
        Randomly draws a mini-batch of examples and their corresponding
        labels.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The number of examples to include in the
            mini-batch.

        Returns
        -------
        sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence examples and their corresponding labels. The
            shape of `sequences` will be
            :math:`B \\times L \\times N`, where :math:`B` is
            `batch_size`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` will be :math:`B \\times F`,
            where :math:`F` is the number of features.

        """
        sequences = np.zeros((batch_size, self.sequence_length, 4))
        targets = np.zeros((batch_size, self.n_features))
        n_samples_drawn = 0
        while n_samples_drawn < batch_size:
            chrom, position = self._sample()
            retrieve_output = self._retrieve(chrom, position)
            if not retrieve_output:
                continue
            seq, seq_targets = retrieve_output
            sequences[n_samples_drawn, :, :] = seq
            targets[n_samples_drawn, :] = seq_targets
            n_samples_drawn += 1
        return (sequences, targets)
