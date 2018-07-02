"""
This module provides the RandomPositionsSampler class.
"""
from collections import namedtuple
import logging
import os
import random

import numpy as np
from pympler import asizeof

from .online_sampler import OnlineSampler
from ..utils import get_indices_and_probabilities

logger = logging.getLogger(__name__)


SampleIndices = namedtuple(
    "SampleIndices", ["indices", "weights"])
"""
TODO: this is common to both the intervals sampler and the
random positions sampler. Can we move this to utils or
somewhere else?

A tuple containing the indices for some samples, and a weight to
allot to each index when randomly drawing from them.

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

    @TODO: possible to generalize to Sequences?
    """

    def __init__(self,
                 reference_sequence,
                 target_path,
                 features,
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
        """
        TODO: rename this to "_partition_by_key"?
        """
        for mode in self.modes:
            self._sample_from_mode[mode] = SampleIndices([], [])
        for index, (chrom, len_chrom) in enumerate(self.reference_sequence.get_chr_lens()):
            if chrom in self.validation_holdout:
                self._sample_from_mode["validate"].indices.append(
                    index)
            elif self.test_holdout and chrom in self.test_holdout:
                self._sample_from_mode["test"].indices.append(
                    index)
            elif '_' not in chrom:  # TODO: remove this.
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
            print(retrieved_seq.shape, chrom, window_start, window_end, strand)
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

    def _update_randcache(self, mode=None):
        if not mode:
            mode = self.mode
        self._randcache[mode]["cache_indices"] = np.random.choice(
            self._sample_from_mode[mode].indices,
            size=200000,
            replace=True,
            p=self._sample_from_mode[mode].weights)
        self._randcache[mode]["sample_next"] = 0

    def sample(self, batch_size=1):
        sequences = np.zeros((batch_size, self.sequence_length, 4))
        targets = np.zeros((batch_size, self.n_features))
        n_samples_drawn = 0
        while n_samples_drawn < batch_size:
            sample_index = self._randcache[self.mode]["sample_next"]
            if sample_index == len(self._randcache[self.mode]["cache_indices"]):
                self._update_randcache()
                sample_index = 0

            rand_interval_index = \
                self._randcache[self.mode]["cache_indices"][sample_index]
            self._randcache[self.mode]["sample_next"] += 1

            interval_info = self.sample_from_intervals[rand_interval_index]
            interval_length = self.interval_lengths[rand_interval_index]

            chrom = interval_info[0]
            position = int(
                interval_info[1] + random.uniform(0, 1) * interval_length)

            retrieve_output = self._retrieve(chrom, position)
            if not retrieve_output:
                continue
            seq, seq_targets = retrieve_output
            sequences[n_samples_drawn, :, :] = seq
            targets[n_samples_drawn, :] = seq_targets
            n_samples_drawn += 1
        return (sequences, targets)

    def get_data_and_targets(self, mode, batch_size, n_samples):
        self.set_mode(mode)
        sequences_and_targets = []

        n_batches = int(n_samples / batch_size)
        for _ in range(n_batches):
            inputs, targets = self.sample(batch_size)
            sequences_and_targets.append((inputs, targets))
        targets_mat = np.vstack([t for (s, t) in sequences_and_targets])
        if mode in self._save_datasets:
            self.save_dataset_to_file(mode, close_filehandle=True)
        return sequences_and_targets, targets_mat

    def get_dataset_in_batches(self, mode, batch_size, n_samples=None):
        if not n_samples and mode == "validate":
            n_samples = 32000
        if not n_samples and mode == "test":
            n_samples = 640000
        return self.get_data_and_targets(mode, batch_size, n_samples)

    def get_validation_set(self, batch_size, n_samples=None):
        return self.get_dataset_in_batches(
            "validate", batch_size, n_samples=n_samples)

    def get_test_set(self, batch_size, n_samples=None):
        return self.get_dataset_in_batches("test", batch_size, n_samples)
