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
        # Holds the size of the parition of each mode (the portion that is
        # specified to belong to each mode)
        self._parititon_sizes = {"validate" : 0, "train" : 0, "test" : 0}

        self._validation_holdout = validation_holdout
        self._N_validation = validation_size

        if test_holdout:
            self._test_holdout = test_holdout
            self._N_test = test_size

        self._N_train = train_size

        # Holds numpy array data sets for each mode (train, validate, test)
        # which is made upon initilization of the class.
        # These samples' initialize size is determined by the sample size
        # designated by the class instance and is subsequently shrunken as
        # calls to sample are made.
        self._partition_ixs = {"train": np.zeros(self._N_train, dtype=np.int64),
                                "validate": np.zeros(self._N_validation, dtype=np.int64),
                                "test": np.zeros(self._N_test, dtype=np.int64)}

        # Count the number of chromosomes and bases. If sampling by chromosome,
        # count the number of base pairs in the chromosomes per mode.
        for chrom, len_chrom in self.reference_sequence.get_chr_lens():
            if not isinstance(validation_holdout, float):
                if chrom in self.validation_holdout:
                    self._parititon_sizes["validate"] += len_chrom
                elif chrom in self.test_holdout:
                    self._parititon_sizes["test"] += len_chrom
                else:
                    self._parititon_sizes["train"] += len_chrom
            self._num_chroms += 1
            self._genome_n_bases += len_chrom

        # Information about each chromosome, "Total" holds the chrom names
        # "Starts" holds the start indices in a theoretical flat array of all
        # chromosomes, "Ends" holds the end indices.
        self.chroms_info = {"Total": [],
                            "Starts": np.zeros(self._num_chroms),
                            "Ends": np.zeros(self._num_chroms)}

        if isinstance(validation_holdout, float):
            self._partition_by_proportion()
        else:
            self._partition_by_chromosome()

    # Set `self.chroms` (list containing chromosome names in order),
    # `self.chroms_starts` (position of each chromosome start in the genome),
    # `self.chroms_ends` (position of each chromosome end in the genome)

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
    # each mode, by proportion (floored to an int)

    def _assign_proportions(self):
        if self.test_holdout:
            test_prop_N = self._genome_n_bases * self._test_holdout
            validation_prop_N = self._genome_n_bases * self._validation_holdout
            training_prop_N = \
                self._genome_n_bases * (1 - self._test_holdout - self._validation_holdout)
            return int(test_prop_N), int(validation_prop_N), int(training_prop_N)
        else:
            validation_prop_N = self._genome_n_bases * self._validation_holdout
            training_prop_N = self._genome_n_bases * (1 - self._validation_holdout)
            return int(validation_prop_N), int(training_prop_N)

    # Should be updated to be dynamic for the genome that is being sampled.
    # If the genome size is not a multiple of 250 million, then we do it
    # differently ..
    def _psuedoshuffle(self):
        total = self._genome_n_bases
        shuffled_sample = np.arange(total, dtype=np.int64)
        jump = 250000000
        num_steps = int(total / jump)
        start = 0
        for i in range(num_steps):
            np.random.shuffle(shuffled_sample[start : start + jump])
            start = start + jump

        start = int(jump / 2)
        for i in range(num_steps - 1):
            np.random.shuffle(shuffled_sample[start : start + jump])
            start = start + jump

        return shuffled_sample

    # Samples for each mode are predetermined based on the sample size for
    # each mode. This imposes an up front time constraint, dependent on each
    # sample size and the size of the entire genome.  # Subsequently, sampling is quick.

    # There is a space constraint in the first phase of picking the samples
    # However, this constraint dwindles after samples are set and even further as
    #  samples are taken and subsequently removed.

    # If the genome is above 500 million base pairs, only a psuedoshuffle is
    # performed on the genome to perform partitions. This is due to time
    # constraints on shuffling larger sets.

    # Samples are taken without replacement. However, if the sample size is
    # larger than the proportion specified, there will be duplicate sample
    # items. If this is not the intended behavior, the sample size should be
    # chosen so as to not be larger than the proportion of the genome.

    def _partition_by_proportion(self):
        self._init_chroms()
        if self._genome_n_bases > 500000000:
            genome_positions_arr = self._psuedoshuffle()
        else:
            genome_positions_arr = np.arange(total, dtype=np.int64)
            np.random.shuffle(genome_positions_arr)

        start = 0
        if self.test_holdout:
            test_prop_N, validation_prop_N, training_prop_N = self._assign_proportions()
            N_test = self._N_test
            while N_test:
                get_N = min(test_prop_N, N_test)
                self._partition_ixs["test"] = genome_positions_arr[start : start + get_N]
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
            N_validation -= get_N
            if N_validation:
                start = start + get_N


        start = start + get_N
        while self._N_train:
            get_N = min(training_prop_N, self._N_train)
            self._partition_ixs["train"] = genome_positions_arr[start:start + get_N]
            self._N_train -= get_N


    def _partition_by_chromosome(self):
        self._init_chroms()
        tot_len, train_counter, validation_counter, test_counter = 0, 0, 0, 0
        # Holds the entire range of values that can be sampled for each mode
        # determined by the user specification of which chrom to sample
        partition = {"validate" : np.zeros(self._parititon_sizes["validate"]),
                        "train" : np.zeros(self._parititon_sizes["train"]),
                        "test" : np.zeros(self._parititon_sizes["test"])}

        for chrom, length in self.reference_sequence.get_chr_lens():

            if chrom in self.validation_holdout:
                partition["validate"][validation_counter: validation_counter + length] = np.arange(tot_len, tot_len + length)
            elif chrom in self.test_holdout:
                partition["test"][test_counter: test_counter + length] = np.arange(tot_len, tot_len + length)
            else:
                partition["train"][train_counter: train_counter + length] = np.arange(tot_len, tot_len + length)
            tot_len += length

        # Given the possibilities to sample from given in `partition`,
        # randomly select `sample_size` elements from `partition` to be in
        # the final sample size for each mode.
        # Sample without replacement unless necessary to do so due to
        # the size of the parition in comparison to the sample size for a mode.
        for mode in self.modes:
            sample_size = self._partition_ixs[mode].size
            counter = 0
            partition_size = self._parititon_sizes[mode]
            while counter < sample_size:
                # If it is possible to sample without replacement, do so.
                if sample_size - counter < partition_size:
                    # This takes a very long time. Maybe pseudoshuffle before
                    # the while loop and take sequentially.
                    self._partition_ixs[mode][counter: sample_size] = np.random.choice(partition[mode],
                                        size=(sample_size - counter), replace=False)
                    counter = sample_size

                # If not (because the user specified more samples than is
                # available in `partition`) shuffle and then add the entire
                # partition to the sample.
                else:
                    # This shuffling may take a long time if the parition is
                    # large. Hopefully would never happen, since the sample size
                    # would be small enough to avoid this.
                    # However, maybe we can use the pseudoshuffle here to
                    # avoid this edge case
                    random.shuffle(partition)
                    self._partition_ixs[mode][counter : counter + partition_size] = partition[mode]
                    counter += partition_size


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
        pos = int(index - self.chroms_info["Starts"][min] - 1)
        return chrom, pos

    def _sample(self):
        random_index = np.random.randint(0, len(self._partition_ixs[self.mode]))
        sample_index = self._partition_ixs[self.mode][random_index]
        chrom, pos = self._pair_from_index(sample_index)
        self._partition_ixs[self.mode] = np.delete(self._partition_ixs[self.mode], random_index)
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
