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
    replacement : boolean, optional
        Default is `True`. Determines if sampling will be done with _sample_with_replacement
        (True) or without replacement (False).
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
                 replacement=True,
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

        self._sample_from_mode = {}
        self._randcache = {}
        for mode in self.modes:
            self._sample_from_mode[mode] = None
            self._randcache[mode] = {"cache_indices": None, "sample_next": 0}

        self.sample_from_intervals = []
        self.interval_lengths = []

        if self._holdout_type == "chromosome":
            self._partition_genome_by_chromosome()
        else:
            self._partition_genome_by_proportion()

        for mode in self.modes:
            self._update_randcache(mode=mode)

        self.replacement = replacement

        self.samples = {}
        for mode, size in [("train", train_size),
            ("validation", validation_size), ("test", test_size)]:
            self.samples[mode] = {"size" : size,
                                  "sample" : np.zeros(size),
                                  "chroms" : np.zeros(0),
                                  "length" : 0 }

        self.sample_init = False

        self.chroms_split = {}
        for chrom_info in ["Total", "Starts", "Ends"]:
            self.chroms_split[chrom_info] = np.zeros(0)

        self.genome_length = 0


    def _partition_genome_by_proportion(self):
        for chrom, len_chrom in self.reference_sequence.get_chr_lens():
            self.sample_from_intervals.append(
                (chrom,
                 self.sequence_length,
                 len_chrom - self.sequence_length))
            self.interval_lengths.append(len_chrom)
        n_intervals = len(self.sample_from_intervals)

        select_indices = list(range(n_intervals))
        np.random.shuffle(select_indices)
        n_indices_validate = int(n_intervals * self.validation_holdout)
        val_indices, val_weights = get_indices_and_probabilities(
            self.interval_lengths, select_indices[:n_indices_validate])
        self._sample_from_mode["validate"] = SampleIndices(
            val_indices, val_weights)

        if self.test_holdout:
            n_indices_test = int(n_intervals * self.test_holdout)
            test_indices_end = n_indices_test + n_indices_validate
            test_indices, test_weights = get_indices_and_probabilities(
                self.interval_lengths,
                select_indices[n_indices_validate:test_indices_end])
            self._sample_from_mode["test"] = SampleIndices(
                test_indices, test_weights)

            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[test_indices_end:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)
        else:
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[n_indices_validate:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)

    def _partition_genome_by_chromosome(self):
        for mode in self.modes:
            self._sample_from_mode[mode] = SampleIndices([], [])
        for index, (chrom, len_chrom) in enumerate(self.reference_sequence.get_chr_lens()):
            if chrom in self.validation_holdout:
                self._sample_from_mode["validate"].indices.append(
                    index)
            elif self.test_holdout and chrom in self.test_holdout:
                self._sample_from_mode["test"].indices.append(
                    index)
            else:
                self._sample_from_mode["train"].indices.append(
                    index)

            self.sample_from_intervals.append(
                (chrom,
                 self.sequence_length,
                 len_chrom - self.sequence_length))
            self.interval_lengths.append(len_chrom - 2 * self.sequence_length)

        for mode in self.modes:
            sample_indices = self._sample_from_mode[mode].indices
            indices, weights = get_indices_and_probabilities(
                self.interval_lengths, sample_indices)
            self._sample_from_mode[mode] = \
                self._sample_from_mode[mode]._replace(
                    indices=indices, weights=weights)

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

    # Returns arrays containing the relative proportions of chromosomes in the
    # train sample, testing sample and validation sample.
    def _get_proportions(self):
        whole_proportions = np.subtract(self.chroms_split["Ends"], self.chroms_split["Starts"])

        # Create masks
        test_mask = np.in1d(self.chroms_split["Total"], self.test_holdout)
        validation_mask = np.in1d(self.chroms_split["Total"], self.validation_holdout)
        holdouts = self.test_holdout + self.validation_holdout
        train_mask = np.in1d(self.chroms_split["Total"], holdouts, invert=True)

        for mode, mask in [("test", test_mask),
                    ("validation", validation_mask),
                    ("train", train_mask)]:
            self.samples[mode]["chroms"] = (self.chroms_split["Total"])[mask]
            self.samples[mode]["proportions"] = \
                [chr_len / self.samples[mode]["length"] for chr_len in whole_proportions[mask]]

        return

    # Update `samples_per_chrom` to map a chromosomes to the number of times it
    # should be sampled in a `sample_size` set, given that each chrom's
    # ratio is given by the `proportions` list. `
    def _samples_from_proportion(self, samples_per_chrom, mode):
        for i in range(self.samples[mode]["size"]):
            chr = np.random.choice(self.samples[mode]["chroms"],
                                   p=self.samples[mode]["proportions"])
            samples_per_chrom[chr] += 1

    # Make a dictionary of chromosomes which maps to how many elements from chrom
    # we sample. Determined probabilistically and weighted by proportion.
    def _samples_per_chrom(self):
        proportions = self._get_proportions()
        samples_per_chrom = defaultdict(int)
        self._samples_from_proportion(samples_per_chrom, "train")
        self._samples_from_proportion(samples_per_chrom, "validation")
        self._samples_from_proportion(samples_per_chrom, "test")
        return samples_per_chrom

    # Setup `self.chroms`, `self.chroms_starts`, `self.chroms_ends`, `self.genome_length`
    def _init_chroms(self):
        tot_len = 0
        for chrom, len_chrom in self.reference_sequence.get_chr_lens():
            self.chroms_split["Total"] = np.append(self.chroms_split["Total"], chrom)
            self.chroms_split["Starts"] = np.append(self.chroms_split["Starts"], tot_len)
            self.chroms_split["Ends"] = np.append(self.chroms_split["Ends"], tot_len + len_chrom)
            tot_len += len_chrom
            if chrom in self.test_holdout:
                self.samples["test"]["length"] += len_chrom
            elif chrom in self.validation_holdout:
                self.samples["validation"]["length"] += len_chrom
            else:
                self.samples["train"]["length"] += len_chrom

        self.genome_length = tot_len

    # Generate a sample for a given chromosome with `samples_per_chrom` elements.
    def _generate_chrom_sample(self, samples_per_chrom, len_chrom, tot_len):
        chrom_range = np.arange(tot_len, tot_len + len_chrom)
        curr_sample = np.random.choice(chrom_range,
                                    size=(1, samples_per_chrom),
                                    replace=False)[0]
        return curr_sample

    # Updates train_sample to hold the indices we will sample from
    # of a hypothetical flattened array of each chromosome laid side by side.
    # These indices can then be converted into (chrom, position) pairs to sample.
    def _generate_sample(self):
        self._init_chroms()

        tot_len = 0
        train_counter = 0
        validation_counter = 0
        test_counter = 0

        samples_per_chrom = self._samples_per_chrom()
        for chrom, len_chrom in self.reference_sequence.get_chr_lens():
            _samples_per_chrom = samples_per_chrom[chrom]
            curr_sample = self._generate_chrom_sample(_samples_per_chrom, len_chrom, tot_len)

            if chrom in self.validation_holdout:
                self.samples["validation"]["sample"][validation_counter  : validation_counter  + _samples_per_chrom] = curr_sample
                validation_counter += _samples_per_chrom
            elif chrom in self.test_holdout:
                self.samples["test"]["sample"][test_counter : test_counter + _samples_per_chrom] = curr_sample
                test_counter += _samples_per_chrom
            else:
                self.samples["train"]["sample"][train_counter  : train_counter  + _samples_per_chrom] = curr_sample
                train_counter += _samples_per_chrom

            tot_len += len_chrom

        self.sample_init = True
        return


    # Returns (chrom, poition) pair corresponding to index in the genome array.
    def _pair_from_index(self, index):
        curr_index = 0
        # find = np.where(self.chrom_ends >= index)
        find = np.where(self.chroms_split["Ends"] >= index)
        next = find[0][0]
        chrom = self.chroms_split["Total"][next]
        pos = int(index - self.chroms_split["Starts"][next])
        return chrom, pos

    def _sample_without_replacement(self):
        if not self.sample_init:
            self._generate_sample()
        random_index = np.random.randint(0, len(self.samples[self.mode]["sample"]))
        sample_index = self.samples[self.mode]["sample"][random_index]
        chrom, pos = self._pair_from_index(sample_index)
        np.delete(self.samples[self.mode]["sample"], random_index)
        return chrom, pos

    def _update_randcache(self, mode=None):
        if not mode:
            mode = self.mode
        self._randcache[mode]["cache_indices"] = np.random.choice(
            self._sample_from_mode[mode].indices,
            size=200000,
            replace=True,
            p=self._sample_from_mode[mode].weights)
        self._randcache[mode]["sample_next"] = 0

    def _sample_with_replacement(self):

        sample_index = self._randcache[self.mode]["sample_next"]
        if sample_index == len(self._randcache[self.mode]["cache_indices"]):
            self._update_randcache()
            sample_index = 0

        rand_interval_index = \
            self._randcache[self.mode]["cache_indices"][sample_index]
        self._randcache[self.mode]["sample_next"] += 1

        chrom, cstart, cend = \
            self.sample_from_intervals[rand_interval_index]
        position = np.random.randint(cstart, cend)

        return chrom, position

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
            if self.replacement:
                chrom, position = self._sample_with_replacement()
            else:
                chrom, position = self._sample_without_replacement()

            retrieve_output = self._retrieve(chrom, position)
            if not retrieve_output:
                continue
            seq, seq_targets = retrieve_output
            sequences[n_samples_drawn, :, :] = seq
            targets[n_samples_drawn, :] = seq_targets
            n_samples_drawn += 1
        return (sequences, targets)
