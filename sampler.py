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

from proteus import Genome
from proteus import GenomicFeatures


LOG = logging.getLogger("deepsea")


class Sampler(object):

    MODES = ("all", "train", "validate", "test")
    STRAND_SIDES = ('+', '-')
    EXPECTED_COLS = (
        "chr", "start", "end")

    def __init__(self, genome,
                 query_feature_data,
                 feature_coordinates,
                 unique_features,
                 chrs_test,
                 chrs_validate,
                 random_seed=436,
                 mode="train"):
        """The class used to sample positive and negative examples from the
        genomic sequence.

        Parameters
        ----------
        genome : str
            Path to the indexed FASTA file of a target organism's complete
            genome sequence.
        query_feature_data : str
            Used for fast querying. Path to tabix-indexed .bed file that
            contains information about genomic features.
            (`genome_features` is the uncompressed original)
        feature_coordinates : str
            Path to a .bed file, uncompressed, that contains the genomic
            coordinates for the features in our dataset. File has the
            columns [chr, start (0-based), end] in order.
            Used for sampling positive examples.
        unique_features : str
            Path to a .txt file containing the set of unique features found in
            our dataset. Each feature is on its own line.
        chrs_test : list[str]
            Specify chromosome(s) to hold out as the test dataset.
            It is expected that the user decides beforehand that the
            proportion of positive examples in the held-out chromosomes
            is appropriate relative to the training and validation datasets.
            e.g. 60-20-20 train-validate-test, 80-10-10, etc.
        chrs_validate : list[str]
            Specify chromosome(s) to hold out as the validation dataset.
            It is expected that the user decides beforehand that the
            proportion of positive examples in the held-out chromosomes
            is appropriate relative to the training and testing datasets.
            e.g. 60-20-20 train-validate-test, 80-10-10, etc.
        random_seed : int, optional
            Default is 436. Sets the numpy random seed.
        mode : {"all", "train", "validate", "test"}, optional
            Default is "train".

        Attributes
        ----------
        genome : Genome
        n_features : int
            Number of unique features in the dataset.
        query_feature_data : GenomicFeatures
        mode : {"all", "train", "validate", "test"}

        Raises
        ------
        ValueError
            - If the input str to `mode` is not one of the specified choices.
        """
        if mode not in self.MODES:
            raise ValueError(
                "Mode must be one of {0}. Input was '{1}'.".format(
                    self.MODES, mode))

        self.genome = Genome(genome)

        t_i = time()
        # used during the positive sampling step - get a random index from the
        # coords file and the corresponding (chr, start, end) info.
        self._coords_df = pd.read_table(
            feature_coordinates, header=None, names=self.EXPECTED_COLS)
        t_f = time()
        LOG.debug(
            ("Loaded genome coordinates for each feature "
             "in the dataset: file {0}, {1} s").format(
                 feature_coordinates, t_f - t_i))

        np.random.seed(random_seed)
        random.seed(random_seed + 1)

        t_i = time()
        features_chr_data = self._coords_df["chr"]
        holdout_chrs_test = np.asarray(features_chr_data.isin(chrs_test))
        holdout_chrs_validate = np.asarray(
            features_chr_data.isin(chrs_validate))
        holdout_chrs_both = np.logical_or(
            holdout_chrs_test, holdout_chrs_validate)

        self._all_indices = features_chr_data.index
        self._training_indices = np.where(~holdout_chrs_both)[0]
        self._test_indices = np.where(holdout_chrs_test)[0]
        self._validate_indices = np.where(holdout_chrs_validate)[0]
        t_f = time()
        LOG.debug(
            ("Partitioned the dataset into train/validate/test sets: "
             "{0} s").format(t_f - t_i))

        self._features = pd.read_csv(unique_features, names=["feature"])
        self._features = self._features["feature"].values.tolist()
        self.n_features = len(self._features)

        self.query_feature_data = GenomicFeatures(
            query_feature_data, self._features)

        self.mode = None
        self.set_mode(mode)  # set the `mode` attribute here.

        LOG.debug("Initialized the Sampler object")

    def set_mode(self, mode):
        """Determines what positive examples are available to sample depending
        on the mode.

        Parameters
        ----------
        mode : {"all", "train", "validate", "test"}
            - all: Use all examples in the genomic features dataset.
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

        if mode == "all":
            indices = self._all_indices
        elif mode == "train":
            indices = self._training_indices
        elif mode == "validate":
            indices = self._validate_indices
        elif mode == "test":
            indices = self._test_indices
        self._use_indices = list(indices)
        self.mode = mode
        LOG.debug("Setting mode to {0}".format(mode))

    def sample_positive(self):
        """Sample a positive example from the genome.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns the sequence encoding and its corresponding feature labels.
        """
        pass

    def sample_negative(self):
        """Sample a negative example from the genome.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns the sequence encoding and an empty array of feature labels
            (since we are sampling a negative example, there should be no
            features present).
        """
        pass

    def sample_mixture(self, positive_proportion=0.50):
        """Gets a mixture of positive and negative samples

        Parameters
        ----------
        positive_proportion : [0.0, 1.0], float, optional
            Default is 0.50. Specify the proportion of positive examples
            that will be sampled (the rest are negative examples).

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns both the sequence encoding and the feature labels
            for the specified range.
        """
        t_i = time()
        sample = self.sample_positive()
        #if random.uniform(0, 1) < positive_proportion:
        #    sample = self.sample_positive()
        #else:
        #    sample = self.sample_negative()
        t_f = time()
        LOG.debug("Sampling step completed: {0} s".format(t_f - t_i))
        return sample


class ChromatinFeaturesSampler(Sampler):

    def __init__(self, genome,
                 query_feature_data,
                 feature_coordinates,
                 unique_features,
                 chrs_test,
                 chrs_validate,
                 radius=100,
                 window_size=1001,
                 random_seed=436,
                 mode="train"):
        """The positive examples we focus on in ChromatinFeaturesSampler are
        those detected in ChIP-seq, DNase-seq, and ATAC-seq experiments; which
        assay DNA binding and accessibility.

        Parameters
        ----------
        radius : int, optional
            Default is 100. The bin is composed of
                ([sequence (len radius)] +
                 position (len 1) + [sequence (len radius)])
            i.e. 201 bp bin.
        window_size : int, optional
            Default is 1001. The input sequence length.
            This should be an odd number to accommodate the fact that
            the bin sequence length will be an odd number.
            i.e. defaults result in 400 bp padding on either side of a
            201 bp bin.

        Attributes
        ----------
        radius : int
        window_size : int
        padding : int
            The amount of padding is identical on both sides

        Raises
        ------
        ValueError
            - If the input `window_size` is less than the computed bin size.
            - If the input `window_size` is an even number.
        """
        super(ChromatinFeaturesSampler, self).__init__(
            genome,
            query_feature_data,
            feature_coordinates,
            unique_features,
            chrs_test,
            chrs_validate,
            random_seed=random_seed,
            mode=mode)

        if window_size < (1 + 2 * radius):
            raise ValueError(
                "Window size of {0} is not greater than bin "
                "size of 1 + 2 x radius {1} = {2}".format(
                    window_size, radius, 1 + 2 * radius))

        if window_size % 2 == 0:
            raise ValueError(
                "Window size must be an odd number. Input was {0}".format(
                    window_size))

        # bin size = self.radius + 1 + self.radius
        self.radius = radius
        self.window_size = window_size
        # the amount of padding is based on the window size and bin size.
        # we use the padding to incorporate sequence context around the
        # bin for which we have feature information.
        self.padding = 0

        remaining_space = window_size - self.radius * 2 - 1
        if remaining_space > 0:
            self.padding = int(remaining_space / 2)

        # used during the negative sampling step - get a random chromosome
        # in the genome FASTA file and randomly select a position in the
        # sequence from there.
        # TODO: negative sampling disabled temporarily to improve training
        # time
        # self._randcache_negatives = self._build_randcache_negatives()

        # used during the positive sampling step - get random indices
        # in the genome FASTA file and randomly select a position within
        # the [start, end) of the genomic coordinates around which we'd
        # define our bin and window.
        self._randcache_positives = self._build_randcache_positives()

        LOG.debug("Initialized the ChromatinFeaturesSampler object")

    def _build_randcache_negatives(self, size=10000):
        t_i = time()
        # select chromosomes
        if "chr" not in self._randcache_negatives \
                or len(self._randcache_negatives["chr"]) == 0:
            rand_chrs = list(np.random.choice(self.genome.chrs, size=size))
            self._randcache_negatives["chr"] = rand_chrs

        # select sequence positions for each chromosome
        if "pos" not in self._randcache_negatives:
            self._randcache_negatives["pos"] = dict(
                [(x, []) for x in self.genome.chrs])
        rand_chr_positions = self._randcache_negatives["pos"]
        for chrom, rand_positions in rand_chr_positions.items():
            if len(rand_positions) == 0:
                rand_chr_positions[chrom] = \
                    self._rand_chr_sequence_positions(
                        self.genome.len_chrs[chrom],
                        size / 2)

        # select strand side
        if "strand" not in self._randcache_negatives \
                or len(self._randcache_negatives["strand"]) == 0:
            rand_strands = list(
                np.random.choice(self.STRAND_SIDES, size=size))
            self._randcache_negatives["strand"] = rand_strands
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
            size=int(size))
        t_f = time()
        LOG.debug(
            ("Built a chromosome-specific cache of random positions: "
             "{0} s").format(t_f - t_i))
        return list(rand_positions)

    def _build_randcache_positives(self, size=10000):
        t_i = time()
        # select examples from all possible examples in the dataset
        if "all" not in self._randcache_positives \
                or len(self._randcache_positives["all"]) == 0:
            randpos_all = list(np.random.choice(self._all_indices, size=size))
            self._randcache_positives["all"] = randpos_all

        # select examples from only the training set
        if "train" not in self._randcache_positives \
                or len(self._randcache_positives["train"]) == 0:
            randpos_train = list(np.random.choice(
                self._training_indices, size=size))
            self._randcache_positives["train"] = randpos_train

        # select examples from only the validation set
        if "validate" not in self._randcache_positives \
                or len(self._randcache_positives["validate"]) == 0:
            randpos_validate = list(np.random.choice(
                self._validate_indices, size=int(size / 2)))
            self._randcache_positives["validate"] = randpos_validate

        # select examples from only the test set
        if "test" not in self._randcache_positives \
                or len(self._randcache_positives["test"]) == 0:
            randpos_test = list(np.random.choice(
                self._test_indices, size=int(size / 2)))
            self._randcache_positives["test"] = randpos_test

        # select strand side
        if "strand" not in self._randcache_positives \
                or len(self._randcache_positives["strand"]) == 0:
            rand_strands = list(np.random.choice(
                self.STRAND_SIDES, size=size))
            self._randcache_positives["strand"] = rand_strands
        t_f = time()
        LOG.info(
            ("Updated the cache for sampling "
             "positive examples: {0} s").format(
                 t_f - t_i))

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
        #LOG.debug("Retrieved ({0}, {1}, {2})".format(
        #    chrom, position, strand))
        sequence_start = position - self.radius - self.padding
        sequence_end = position + self.radius + self.padding + 1
        retrieved_sequence = \
            self.genome.get_encoding_from_coords(
                chrom, sequence_start, sequence_end, strand)
        bin_start = position - self.radius
        bin_end = position + self.radius + 1
        if not is_positive:
            return (
                retrieved_sequence,
                np.zeros((bin_end - bin_start,
                         self.query_feature_data.n_features)))
        else:
            retrieved_data = self.query_feature_data.get_feature_data(
                chrom, bin_start, bin_end, strand)
            return (retrieved_sequence, retrieved_data)

    def sample_negative(self):
        """Sample a negative example from the genome.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns the sequence encoding for the window and an array
            of all 0s (no features) for the middle bin.
        """
        if len(self._randcache_negatives) == 0 or \
                len(self._randcache_negatives["chr"]) == 0:
            self._build_randcache_negatives()
        randchr = self._randcache_negatives["chr"].pop()

        if len(self._randcache_negatives["pos"][randchr]) == 0:
            t_i = time()
            self._randcache_negatives["pos"][randchr] = \
                self._rand_chr_positions(
                    self.genome.len_chrs[randchr], 500)
            t_f = time()
            LOG.debug(
                ("Updated the cache for sampling negative examples, "
                 "{0} positions only: {1} s").format(randchr, t_f - t_i))
        randpos = self._randcache_negatives["pos"][randchr].pop()

        if len(self._randcache_negatives["strand"]) == 0:
            self._build_randcache_negatives()
        randstrand = self._randcache_negatives["strand"].pop()

        is_positive = self.query_feature_data.is_positive(
            randchr, randpos - self.radius, randpos + self.radius + 1)
        if is_positive:
            LOG.debug(
                "Sample background overlapped with positive examples. "
                "Trying `sample_negative` again.")
            return self.sample_negative()
        else:
            return self._retrieve(randchr, randpos, randstrand,
                                  is_positive=False)

    def sample_positive(self):
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
                len(self._randcache_positives[self.mode]) == 0 or \
                len(self._randcache_positives["strand"]) == 0:
            self._build_randcache_positives()
        randindex = self._randcache_positives[self.mode].pop()
        row = self._coords_df.iloc[randindex]

        gene_length = row["end"] - row["start"]
        chrom = row["chr"]

        rand_in_gene = random.uniform(0, 1) * gene_length
        position = int(
            row["start"] + rand_in_gene)

        # we have verified that there is no strand information
        # in any of our data
        strand = self._randcache_positives["strand"].pop()

        seq, feats = self._retrieve(chrom, position, strand,
                                    is_positive=True)
        n, k = seq.shape
        if n == 0:
            LOG.debug(
                "Sample positive window was out of bounds. "
                "Trying `sample_positive` again.")
            return self.sample_positive()
        else:
            return (seq, feats)