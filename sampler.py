"""This class samples from the dataset of genomic coordinates, corresponding
to the features (genomic, chromatin) detected in an organism's genome.
It has both the Genome and GenomicFeatures objects as well as sampling
functions that draw both positive and negative sequence examples from the
genome.
"""
import logging
import random
from time import time

import numpy as np
import pandas as pd
from pybedtools import BedTool

from data_utils import Genome
from data_utils import GenomicFeatures


LOG = logging.getLogger("deepsea")


class Sampler(object):

    SAMPLING_OPTIONS = ("random", "positive", "proportion")
    STRAND_SIDES = ('+', '-')

    def __init__(self,
                 genome,
                 query_feature_data,
                 unique_features,
                 random_seed=436,
                 sample_from="random",
                 sample_positive_prop=0.50):
        """The class used to sample positive and negative examples from the
        genomic sequence.

        Parameters
        ----------
        genome : str
            Path to the indexed FASTA file of a target organism's complete
            genome sequence.
        query_feature_data : str
            Used for fast querying. Path to tabix-indexed .bed.gz file that
            contains information about genomic features.
        unique_features : str
            Path to a .txt file containing the set of unique features found in
            our dataset. Each feature is on its own line.
        random_seed : int, optional
            Default is 436. Sets the numpy random seed.
        sample_from : {"random", "positive", "proportion"}, optional
            Default is "random". Specify where to draw our samples.
            * "random": Select a random chromosome, sequence position in the
                  chr, and strand side. Do not check for whether it is a known
                  positive example (i.e. has at least 1 genomic feature
                  within the surrounding bin).
            * "positive":
                  The samples we draw will have at least 1 genomic feature
                  as a result.
            * "proportion": Select from the positive examples x%
                  of the time and select a negative example
                  (0 genomic features in this bin) (100 - x)% of the time.
        sample_positive_prop : [0.0, 1.0], optional
            Default is 0.50. This is only used if `sample_from` is
            "proportion". x = `sample_positive_prop` * 100%

        Attributes
        ----------
        genome : Genome
        n_features : int
            Number of unique features in the dataset.
        query_feature_data : GenomicFeatures
        sample_from : {"random", "positive", "proportion"}
        sample_positive_prop : [0.0, 1.0]

        Raises
        ------
        ValueError
            - If the input str to `sample_from` is not one of the specified
              choices.
        """
        if sample_from not in self.SAMPLING_OPTIONS:
            raise ValueError(
                ("Sampling option `sample_from` must be one of {0}. "
                 "Input was '{1}'.").format(
                     self.SAMPLING_OPTIONS, sample_from))

        self.genome = Genome(genome)

        # set the necessary random seeds.
        np.random.seed(random_seed)
        random.seed(random_seed + 1)

        self._features = []
        with open(unique_features, "r") as file_handle:
            for line in file_handle:
                self._features.append(line.strip())
        self.n_features = len(self._features)

        self.query_feature_data = GenomicFeatures(
            query_feature_data, self._features)

        self.sample_from = sample_from
        self.sample_positive_prop = sample_positive_prop

        LOG.debug("Initialized the Sampler object")

    def get_feature_from_index(self, feature_index):
        """Returns the feature corresponding to an index in the feature
        vector. Currently used by the 'deepsea' logger.

        Parameters
        ----------
        feature_index : int

        Returns
        -------
        str
        """
        return self.query_feature_data.index_feature_map[feature_index]

    def get_sequence_from_encoding(self, encoding):
        return self.genome.encoding_to_sequence(encoding)

    def sample(self, sample_batch=1):
        """Sample based on the `self.sample_from` value specified during
        initialization of the Sampler object.

        Returns
        -------
        @TODO: change this if we keep the `sample_batch` concept.
        tuple(np.ndarray, np.ndarray)
            Returns the sequence encoding and its corresponding feature labels.
        """
        sample = None
        if self.sample_from == "random":
            sample = self.sample_random(sample_batch)
        elif self.sample_from == "positive":
            sample = self.sample_positive(sample_batch)
        elif self.sample_from == "proportion":
            sample = self.sample_mixture(sample_batch)
        return sample

    def sample_random(self, sample_batch):
        """Sample an example from the genome. This should be implemented
        such that there is no need to check whether the drawn example is
        positive or negative.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns the sequence encoding and its corresponding feature labels.
        """
        pass

    def sample_negative(self, sample_batch):
        """Sample a negative example from the genome.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns the sequence encoding and an empty array of feature labels
            (since we are sampling a negative example, we expect no
            features to be present).
        """
        pass

    def sample_positive(self, sample_batch):
        """Sample a positive example from the genome.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns the sequence encoding and its corresponding feature labels.
        """
        pass

    def sample_mixture(self, sample_batch):
        """Gets a mixture of positive and negative samples, where the
        frequency of each being drawn is determined by the
        `self.sample_positive_prop` parameter.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns both the sequence encoding and the feature labels
            for the specified range.
        """
        sample = None
        if random.uniform(0, 1) < self.sample_positive_prop:
            sample = self.sample_positive(sample_batch)
        else:
            sample = self.sample_negative(sample_batch)
        return sample


class ChromatinFeaturesSampler(Sampler):

    MODES = ("train", "validate", "test")

    def __init__(self,
                 genome,
                 query_feature_data,
                 feature_coordinates,
                 unique_features,
                 chrs_test,
                 n_validate_from_train=8000,
                 window_size=1001,
                 bin_radius=100,
                 bin_feature_threshold=0.5,
                 mode="train",
                 random_seed=436,
                 sample_from="random",
                 sample_positive_prop=0.5):
        """The positive examples we focus on in ChromatinFeaturesSampler are
        those detected in ChIP-seq, DNase-seq, and ATAC-seq experiments; which
        assay DNA binding and accessibility.

        Parameters
        ----------
        feature_coordinates : str
            Path to a .bed file that contains the genomic
            coordinates for the features in our dataset. File has the
            columns [chr, start (0-based), end] in order.
            Used for sampling positive examples.
        chrs_test : list[str]
            Specify chromosome(s) to hold out as the test dataset.
            It is expected that the user determines beforehand that the
            proportion of positive examples in the held-out chromosomes
            is appropriate relative to the training and validation datasets.
        n_validate_from_train : int
            Default is 8000. The number of examples to hold out from the
            training set. These examples are our validation dataset and
            are used to evaluate the model at the end of each training epoch.
        window_size : int, optional
            Default is 1001. The input sequence length.
            This should be an odd number to accommodate the fact that
            the bin sequence length will be an odd number.
            i.e. default results in 400 bp padding on either side of a
            201 bp bin.
        bin_radius : int, optional
            Default is 100. The bin is composed of
                ([sequence (len radius)] +
                 position (len 1) + [sequence (len radius)])
            i.e. bin size of 201 bp.
        bin_feature_threshold : float, optional
            Default is 0.50. The minimum length of a chromatin feature peak
            detected within a bin must be greater than or equal to
            `bin_feature_threshold` * the bin size.
        mode : {"train", "validate", "test"}, optional
            Default is "train".

        Attributes
        ----------
        radius : int
        window_size : int
        padding : int
            The amount of padding is identical on both sides
        bin_feature_threshold : float
        chrs_test : list(str), e.g. ["chr8", "chr9"]
        n_validation : int
        mode : {"train", "validate", "test"}

        Raises
        ------
        ValueError
            - If the input str to `mode` is not one of the specified choices.
            - If the input `window_size` is less than the computed bin size.
            - If the input `window_size` is an even number.
        """
        super(ChromatinFeaturesSampler, self).__init__(
            genome,
            query_feature_data,
            unique_features,
            random_seed=random_seed,
            sample_from=sample_from,
            sample_positive_prop=sample_positive_prop)

        if mode not in self.MODES:
            raise ValueError(
                "Mode must be one of {0}. Input was '{1}'.".format(
                    self.MODES, mode))

        if window_size < (1 + 2 * bin_radius):
            raise ValueError(
                "Window size of {0} is less than the bin "
                "size of 1 + 2 x radius {1} = {2}".format(
                    window_size, bin_radius, 1 + 2 * bin_radius))

        if window_size % 2 == 0:
            raise ValueError(
                "Window size must be an odd number. Input was {0}".format(
                    window_size))

        # total bin size = self.radius + 1 + self.radius
        self.radius = bin_radius
        self.window_size = window_size
        # the amount of padding is based on the window size and bin size.
        # we use the padding to incorporate sequence context around the
        # bin for which we have feature information.
        self.padding = 0

        remaining_space = window_size - self.radius * 2 - 1
        if remaining_space > 0:
            self.padding = int(remaining_space / 2)

        self.bin_feature_threshold = bin_feature_threshold

        self.chrs_test = chrs_test
        self.n_validation = n_validate_from_train

        self._load_feature_coordinates(feature_coordinates)
        self._partition_dataset()

        # used during the negative sampling step - get a random chromosome
        # in the genome FASTA file and randomly select a position in the
        # sequence from there.
        if self.sample_from == "random" or self.sample_from == "proportion":
            self._randcache_background = {}
            self._build_randcache_background()

        # used during the positive sampling step - get random indices
        # in the genome FASTA file and randomly select a position within
        # the [start, end) of the genomic coordinates around which we'd
        # define our bin and window.
        self._randcache_positives = {}
        self._build_randcache_positives()

        self.mode = None
        self.set_mode(mode)  # set mode in this function

        LOG.debug("Initialized the ChromatinFeaturesSampler object")

    def _load_feature_coordinates(self, feature_coordinates_file):
        """Used during the positive sampling step. Gets a random row index from
        the coordinates file and the corresponding (chr, start, end)
        information.
        """
        t_i = time()
        tmp_coords_df = pd.read_table(
            feature_coordinates_file,
            header=None,
            names=("chrom", "start", "end"))

        holdout_chrs_test = np.asarray(
            tmp_coords_df["chrom"].isin(self.chrs_test))
        not_test_indices = list(np.where(~holdout_chrs_test)[0])

        self._validation_indices = list(
            np.random.choice(
                not_test_indices, size=self.n_validation, replace=False))

        validation_rows = tmp_coords_df.index.isin(self._validation_indices)
        self._validation_df = tmp_coords_df[validation_rows]

        tmp_coords_df = tmp_coords_df[~validation_rows]

        coords_bedtool = BedTool().from_dataframe(tmp_coords_df)

        merged_intervals = coords_bedtool.merge()

        self._coords_df = merged_intervals.to_dataframe()

        self._coords_df["interval_length"] = self._coords_df["end"].sub(
            self._coords_df["start"], axis=0)

        min_feature_size = (self.radius * 2 + 1) * self.bin_feature_threshold
        self._coords_df = self._coords_df[
            self._coords_df["interval_length"] >= min_feature_size]

        t_f = time()
        LOG.debug(
            ("Loaded genome coordinates for each feature "
             "in the dataset: file {0}, {1} s").format(
                 feature_coordinates_file, t_f - t_i))

    def _partition_dataset(self):
        """Specify the training, validation, and test indices available to
        sample in the genomic features coordinates dataframe.
        """
        t_i = time()
        features_chr_data = self._coords_df["chrom"]
        holdout_chrs_test = np.asarray(
            features_chr_data.isin(self.chrs_test))

        self._training_indices = list(np.where(~holdout_chrs_test)[0])
        self._test_indices = list(np.where(holdout_chrs_test)[0])

        self._training_weights = \
            self._coords_df.ix[self._training_indices]["interval_length"] \
                .tolist()
        remove_indices = np.argwhere(np.isnan(self._training_weights))
        self._training_indices = [
            ti for i, ti in enumerate(self._training_indices) if i not in remove_indices]

        self._training_weights = [
            tw for i, tw in enumerate(self._training_weights) if i not in remove_indices]
        self._training_weights = \
            np.array(self._training_weights) / float(np.sum(self._training_weights))

        self._test_weights = \
            self._coords_df.ix[self._test_indices]["interval_length"] \
                .tolist()
        remove_indices = np.argwhere(np.isnan(self._test_weights))
        self._test_indices = [
            ti for i, ti in enumerate(self._test_indices) if i not in remove_indices]

        self._test_weights = [
            tw for i, tw in enumerate(self._test_weights) if i not in remove_indices]
        self._test_weights = \
            np.array(self._test_weights) / float(np.sum(self._test_weights))

        t_f = time()
        LOG.debug(
            ("Partitioned the dataset into train/validate & test sets: "
             "{0} s").format(t_f - t_i))

    def set_mode(self, mode):
        """Determines what positive examples are available to sample depending
        on the mode.

        Parameters
        ----------
        mode : {"train", "validate", "test"}
            - train: Use all examples except those in the test and validation
                     holdout sets.
            - test: Use only the examples in the test holdout chromosome set.
            - validation: Use only the examples in the validation holdout
                          chromosome set.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the input str to `mode` is not one of the specified choices.
        """
        if mode not in self.MODES:
            raise ValueError(
                "Mode must be one of {0}. Input was '{1}'.".format(
                    self.MODES, mode))

        self.mode = mode
        LOG.debug("[MODE] {0}".format(mode))

    def _retrieve(self, chrom, position, strand,
                  is_positive=False):
        """Retrieve the sequence surrounding the given position in a
        chromosome's sequence for a specific strand side. The amount of
        context (surrounding sequence) provided for that position
        is set by the `self.window_size` attribute.

        Parameters
        ----------
        chrom : str
            e.g. "chr1".
        position : int
        strand : {'+', '-'}
        is_positive : bool, optional
            Default is True. If the chromosome sequence position is verified
            to be a positive example, query and return the feature labels
            as well.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            If not `is_positive`, returns the sequence encoding and a numpy
            array of zeros (no feature labels present).
            Otherwise, returns both the sequence encoding and the feature
            labels for the specified range.
        """
        bin_start = position - self.radius
        bin_end = position + self.radius + 1
        retrieved_targets = self.query_feature_data.get_feature_data(
            chrom, bin_start, bin_end)

        if is_positive and np.sum(retrieved_targets) == 0:
            return (np.zeros((0, 4)), retrieved_targets)

        window_start = bin_start - self.padding
        window_end = bin_end + self.padding
        retrieved_sequence = \
            self.genome.get_encoding_from_coords(
                chrom, window_start, window_end, strand)
        return (retrieved_sequence, retrieved_targets)

    def _get_rand_background(self):
        if len(self._randcache_background) == 0 or \
                len(self._randcache_background["chr"]) == 0:
            self._build_randcache_background()

        randchr = self._randcache_background["chr"].pop()

        if len(self._randcache_background["pos"][randchr]) == 0:
            t_i = time()
            self._randcache_background["pos"][randchr] = \
                self._rand_chr_sequence_positions(
                    self.genome.len_chrs[randchr], 500)
            t_f = time()
            LOG.debug(
                ("[RANDCACHE] Updated the cache for sampling negative "
                 "examples, {0} positions only: {1} s").format(
                    randchr, t_f - t_i))

        randpos = self._randcache_background["pos"][randchr].pop()
        randstrand = self.STRAND_SIDES[random.randint(0, 1)]
        return (randchr, randpos, randstrand)

    def sample_random(self, sample_batch):
        """Sample an example from the genome. This should be implemented
        such that there is no need to check whether the drawn example is
        positive or negative.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns the sequence encoding and its corresponding feature labels.
        """
        randchr, randpos, randstrand = self._get_rand_background()
        return self._retrieve(randchr, randpos, randstrand,
                              is_positive=False)

    def sample_negative(self, sample_batch):
        """Sample a negative example from the genome.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns the sequence encoding for the window and an array
            of all 0s (no features) for the middle bin.
        """
        randchr, randpos, randstrand = self._get_rand_background()
        is_positive = self.query_feature_data.is_positive(
            randchr, randpos - self.radius, randpos + self.radius + 1)
        while is_positive:
            LOG.debug(
                "Sample background overlapped with positive examples. "
                "Trying `sample_negative` again.")
            randchr, randpos, randstrand = self._get_rand_background()
            is_positive = self.query_feature_data.is_positive(
                randchr, randpos - self.radius, randpos + self.radius + 1)
        else:
            return self._retrieve(randchr, randpos, randstrand,
                                  is_positive=False)

    def _sample_positive(self):
        if len(self._randcache_positives) == 0 or \
                len(self._randcache_positives[self.mode]) == 0:
            self._build_randcache_positives(size=64000)
        randindex = self._randcache_positives[self.mode].pop()
        row = None
        if self.mode == "validate":
            row = self._validation_df.ix[randindex]
        else:
            row = self._coords_df.iloc[randindex]

        interval_length = row["end"] - row["start"]
        chrom = row["chrom"]
        position = int(row["start"] + random.uniform(0, 1) * interval_length)
        strand = self.STRAND_SIDES[random.randint(0, 1)]
        seq, feats = self._retrieve(chrom, position, strand,
                                    is_positive=True)
        return (seq, feats)

    def sample_positive(self, sample_batch):
        """Sample a positive example from the genome.

        Parameters
        ----------

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns both the sequence encoding for the window and
            the feature labels for the middle bin.
        """
        if len(self._randcache_positives) == 0 or \
                len(self._randcache_positives[self.mode]) == 0:
            self._build_randcache_positives(size=64000)
        sequences = np.zeros((sample_batch, self.window_size, 4))
        targets = np.zeros((sample_batch, self.n_features))
        for i in range(sample_batch):
            seq, feats = self._sample_positive()
            while seq.shape[0] == 0 or \
                    np.sum(seq) / float(seq.shape[0]) < 0.70:
                if seq.shape[0] == 0:
                    LOG.debug(
                        "Sample positive was out of bounds. Trying again.")
                else:
                    LOG.debug(
                        ("Too many unknowns in the retrieved sequence. "
                         "Trying again."))
                seq, feats = self._sample_positive()
            sequences[i, :, :] = seq
            targets[i, :] = feats
        return (sequences, targets)

    def _build_randcache_background(self, size=10000):
        # TODO: this needs to change if we decide to sample in batches
        t_i = time()
        # select chromosomes
        if "chr" not in self._randcache_background \
                or len(self._randcache_background["chr"]) == 0:
            rand_chrs = list(np.random.choice(self.genome.chrs, size=size))
            self._randcache_background["chr"] = rand_chrs

        # select sequence positions for each chromosome
        if "pos" not in self._randcache_background:
            self._randcache_background["pos"] = dict(
                [(x, []) for x in self.genome.chrs])
        rand_chr_positions = self._randcache_background["pos"]
        for chrom, rand_positions in rand_chr_positions.items():
            if len(rand_positions) == 0:
                rand_chr_positions[chrom] = \
                    self._rand_chr_sequence_positions(
                        self.genome.len_chrs[chrom],
                        int(size / 3))
        t_f = time()
        LOG.info(
            ("Updated the cache for sampling "
             "negative examples: {0} s").format(
                 t_f - t_i))

    def _rand_chr_sequence_positions(self, chr_len, size):
        t_i = time()
        rand_positions = np.random.choice(range(
            self.radius + self.padding,
            chr_len - self.radius - self.padding - 1),
            size=size)
        t_f = time()
        LOG.debug(
            ("Built a chromosome-specific cache of random positions: "
             "{0} s").format(t_f - t_i))
        return list(rand_positions)

    def _build_randcache_positives(self, size=None):
        t_i = time()
        # select examples from only the training set
        if "train" not in self._randcache_positives \
                or len(self._randcache_positives["train"]) == 0:
            randpos_train = list(np.random.choice(
                self._training_indices,
                size=size if size else len(self._training_indices),
                p=self._training_weights,
                replace=False))
            self._randcache_positives["train"] = randpos_train

        # select examples from only the test set
        if "test" not in self._randcache_positives \
                or len(self._randcache_positives["test"]) == 0:
            randpos_test = list(np.random.choice(
                self._test_indices,
                size=size if size else len(self._test_indices),
                p=self._test_weights,
                replace=False))
            self._randcache_positives["test"] = randpos_test

        # select examples from only the validation set
        if "validate" not in self._randcache_positives \
                or len(self._randcache_positives["validate"]) == 0:
            validation_shuffled = random.sample(
                self._validation_indices, len(self._validation_indices))
            self._randcache_positives["validate"] = validation_shuffled

        t_f = time()
        LOG.info(
            ("Updated the cache for sampling "
             "positive examples: {0} s").format(
                 t_f - t_i))
