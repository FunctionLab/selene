"""TODO: this class is not functional yet. Please do not use/review
at this time.
"""
from collections import namedtuple
import random

import numpy as np

from .online_sampler import OnlineSampler


SampleIndices = namedtuple(
    "SampleIndices", ["indices", "weights"])


def _get_indices_and_probabilities(interval_lengths, indices):
    interval_lens = np.array(interval_lengths)[indices]
    weights = interval_lens / float(np.sum(interval_lens))

    keep_indices = []
    for index, weight in enumerate(weights):
        if weight > 1e-10:
            keep_indices.append(indices[index])
    if len(keep_indices) == len(indices):
        assert len(indices) == len(weights.tolist())
        return indices, weights.tolist()
    else:
        return _get_indices_and_probabilities(
            interval_lengths, keep_indices)


class RandomSampler(OnlineSampler):

    def __init__(self,
                 genome,
                 query_feature_data,
                 distinct_features,
                 random_seed=436,
                 validation_holdout=['6', '7'],
                 test_holdout=['8', '9'],
                 sequence_length=1001,
                 center_bin_to_predict=201,
                 feature_thresholds=0.5,
                 mode="train"):
        super(RandomSampler, self).__init__(
            genome,
            query_feature_data,
            distinct_features,
            random_seed=random_seed,
            validation_holdout=validation_holdout,
            test_holdout=test_holdout,
            sequence_length=sequence_length,
            center_bin_to_predict=center_bin_to_predict,
            feature_thresholds=feature_thresholds,
            mode=mode)

        self.sample_from_mode = {}
        self.randcache = {}
        for mode in self.modes:
            self.sample_from_mode[mode] = None
            self.randcache[mode] = {"cache": None, "sample_next": 0}

        if self._holdout_type == "chromosome":
            self._partition_genome_by_chromosome()
        else:
            self._partition_genome_by_proportion()

        self._update_randcache()

    def _partition_genome_by_proportion(self):
        # what is the diff between a randcache that
        # randomly selects a position from all
        # possible positions in chromosomes
        # vs one that selects the chromosome first
        # and then the index?
        total_positions = 0
        chrom_up_to_position = {}
        for chrom, len_chrom in self.genome.get_chr_lens():
            total_positions += len_chrom
            chrom_up_to_position[chrom] = total_positions
        # divide total_positions into 10 segments.


        select_indices = list(range(len(self.sample_from_intervals)))
        np.random.shuffle(select_indices)
        n_indices_validate = int(
            len(self.sample_from_intervals) * self.validation_holdout)
        val_indices, val_weights = _get_indices_and_probabilities(
            self.interval_lengths, select_indices[:n_indices_validate])
        self.sample_from_mode["validate"] = SampleIndices(
            val_indices, val_weights)

        if self.test_holdout:
            n_indices_test = int(
                len(self.sample_from_intervals) * self.test_holdout)
            test_indices_end = n_indices_test + n_indices_validate
            test_indices, test_weights = _get_indices_and_probabilities(
                self.interval_lengths,
                select_indices[n_indices_validate:test_indices_end])
            self.sample_from_mode["test"] = SampleIndices(
                test_indices, test_weights)

            tr_indices, tr_weights = _get_indices_and_probabilities(
                self.interval_lengths, select_indices[test_indices_end:])
            self.sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)
        else:
            tr_indices, tr_weights = _get_indices_and_probabilities(
                self.interval_lengths, select_indices[n_indices_validate:])
            self.sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)

    def _partition_genome_by_chromosome(self, intervals_file):
        training_chrs = self.genome.get_chrs()
        training_chrs = set(training_chrs) - set(self.validation_holdout)
        if self.test_holdout:
            training_chrs = list(
                training_chrs - set(self.test_holdout))
        else:
            training_chrs = list(training_chrs)

        for mode in self.modes:
            self.sample_from_mode[mode] = SampleIndices([], [])
        with open(intervals_file, 'r') as file_handle:
            for index, line in enumerate(file_handle):
                cols = line.strip().split('\t')
                chrom = cols[0]
                start = int(cols[1])
                end = int(cols[2])
                if chrom in self.validation_holdout:
                    self.sample_from_mode["validate"].indices.append(
                        index)
                elif self.test_holdout and chrom in self.test_holdout:
                    self.sample_from_mode["test"].indices.append(
                        index)
                else:
                    self.sample_from_mode["train"].indices.append(
                        index)
                self.sample_from_intervals.append(
                    (chrom, start, end))
                self.interval_lengths.append(end - start)

        for mode in self.modes:
            sample_indices = self.sample_from_mode[mode].indices
            indices, weights = _get_indices_and_probabilities(
                self.interval_lengths, sample_indices)
            self.sample_from_mode[mode] = \
                self.sample_from_mode[mode]._replace(
                    indices=indices, weights=weights)

    def _retrieve(self, chrom, position, strand):
        bin_start = position - self.bin_radius
        bin_end = position + self.bin_radius + 1
        retrieved_targets = self.query_feature_data.get_feature_data(
            chrom, bin_start, bin_end)
        if np.sum(retrieved_targets) == 0:
            return None

        window_start = bin_start - self.surrounding_sequence_radius
        window_end = bin_end + self.surrounding_sequence_radius
        retrieved_sequence = \
            self.genome.get_encoding_from_coords(
                "chr{0}".format(chrom), window_start, window_end, strand)
        return (retrieved_sequence, retrieved_targets)

    def _update_randcache(self):
        mode = self.mode
        self.randcache[mode]["cache_indices"] = np.random.choice(
            self.sample_from_mode[mode].indices,
            size=len(self.sample_from_mode[mode].indices),
            replace=True,
            p=self.sample_from_mode[mode].weights)
        self.randcache[mode]["sample_next"] = 0
        return

    def sample(self, batch_size):
        sequences = np.zeros((batch_size, self.sequence_length, 4))
        targets = np.zeros((batch_size, self.n_features))
        n_samples_drawn = 0
        while n_samples_drawn < batch_size:
            sample_index = self.randcache[self.mode]["sample_next"]
            if sample_index == len(self.sample_from_mode[self.mode].indices):
                self._update_randcache()
            rand_interval_index = \
                self.randcache[self.mode]["cache_indices"][sample_index]
            self.randcache[self.mode]["sample_next"] += 1
            interval_info = self.sample_from_intervals[rand_interval_index]
            interval_length = self.interval_lengths[rand_interval_index]
            chrom = interval_info[0]
            position = int(
                interval_info[1] + random.uniform(0, 1) * interval_length)
            strand = self.STRAND_SIDES[random.randint(0, 1)]

            retrieve_output = self._retrieve(chrom, position, strand)
            if not retrieve_output:
                continue
            seq, seq_targets = retrieve_output
            if seq.shape[0] == 0 or np.sum(seq) / float(seq.shape[0]) < 0.70:
                continue
            sequences[n_samples_drawn, :, :] = seq
            targets[n_samples_drawn, :] = seq_targets
            n_samples_drawn += 1
        return (sequences, targets)
