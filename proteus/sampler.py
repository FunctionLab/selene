"""This class samples from the dataset of genomic coordinates, corresponding
to the features (genomic, chromatin) detected in an organism's genome.
It has both the Genome and GenomicFeatures objects as well as sampling
functions that draw both positive and negative sequence examples from the
genome.
"""
import numpy as np
import pandas as pd

import Genome
import GenomicFeatures

class Sampler(object):

    MODES = ("all", "train", "validate", "test")
    EXPECTED_BED_COLS = (
        "chr", "start", "end", "strand", "feature") # "metadata_index")
    USE_BED_COLS = (
        "chr", "start", "end", "strand")
    STRAND_SIDES = ('+', '-')

    def __init__(self, genome, genomic_features, query_features,
                 chrs_test, validation_prop=0.2,
                 radius=100, window_size=1001,
                 random_seed=436, mode="all"):
        """The class used to sample positive and negative examples from the
        genomic sequence. These examples are used during training/testing
        of the model.

        Parameters
        ----------
        genome : str
            Path to the indexed FASTA file of a target organism's complete
            genome sequence.
        genomic_features : str
            Path to the .bed file that contains information about
            genomic features.
            File must have the following columns, in order:
                [chr, start (0-based), end, strand, feature, metadata_index]
        query_features : str
            Used for fast querying. Path to tabix-indexed .bed file that
            contains information about genomic features.
            (`genome_features` is the uncompressed original)
        chrs_test : list[str]
            Specify chromosomes to hold out as the test dataset.
        validation_prop : float, optional
            Default is 0.2. The proportion of positive examples to hold out
            from our training dataset as the validation dataset.
            Selects `validation_prop` * 100% indices from `genomic_features`
            to exclude from training.
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
        random_seed : int, optional
            Default is 436. Sets the numpy random seed.
        mode : {"all", "train", "test"}, optional
            Default is "all".

        Attributes
        ----------
        genome : Genome
        query_features : GenomicFeatures
        radius : int
        padding : int
            Should be identical on both sides
        mode : {"all", "train", "test"}

        Raises
        ------
        ValueError
            - If the input str to `mode` is not one of the specified choices.
            - If the input `window_size` is less than the computed bin size.
            - If the input `window_size` is an even number.
        """
        if mode not in self.MODES:
            raise ValueError(
                "Mode must be one of {0}. Input was '{1}'.".format(
                    self.MODES, mode))

        if window_size < (1 + 2 * radius):
            raise ValueError(
                "Window size of {0} is not greater than bin "
                "size of 1 + 2 x radius {1} = {2}".format(
                    window_size, radius, 1 + 2 * radius))


        if window_size % 2 == 0:
            raise ValueError(
                "Window size must be an odd number. Input was {0}".format(
                    window_size))

        self.genome = Genome(genome)

        # used during the positive sampling step - get a random index from the
        # .bed file and the corresponding (chr, start, end, strand).
        self._features_df = pd.read_table(
            genomic_features, header=None, names=self.EXPECTED_BED_COLS,
            usecols=self.USE_BED_COLS)
        # stores a copy of the .bed file that can be used to reset
        # `self._features_df` depending on what mode is specified.
        self._dup_features_df = pd.read_table(
            genomic_features, header=None, names=self.EXPECTED_BED_COLS,
            usecols=self.USE_BED_COLS)

        self._test_indices = self._features_df["chr"].isin(test_chrs)
        self._training_indices = ~self._features_df["chr"].isin(test_chrs)
        self._validation_indices = np.random.choice(
            self._training_indices,
            size=len(self._training_indices) * validation_prop,
            replace=False)
        validation_set = set(self._validation_indices)
        self._training_indices = [ix for ix in self._training_indices
                                  if ix not in validation_set]

        features = self._features_df["feature"].unique()
        self.n_features = len(features)
        self.query_features = GenomicFeatures(query_features, features)

        # bin size = self.radius + 1 + self.radius
        self.radius = radius
        # the amount of padding is based on the window size and bin size.
        # we use the padding to incorporate sequence context around the
        # bin for which we have feature information.
        self.padding = 0

        remaining_space = window_size - self.radius * 2 - 1
        if remaining_space > 0:
            self.padding = int(remaining_space / 2)

        self.set_mode(mode)

        np.random.seed(random_seed)

        # used during the background sampling step - get a random chromosome
        # in the genome FASTA file and randomly select a position in the
        # sequence from there.
        self._randcache = []

    def set_mode(self, mode):
        """Determines what positive examples are available to sample depending
        on the mode.

        Parameters
        ----------
        mode : {"all", "train", "test}
            - all:   Use all examples in the genomic features dataset.
            - train: Use all examples except those in the holdout
                     chromosome set.
            - test:  Use only the examples in the holdout chromosome set.
        """
        if mode == "all":
            self._features_df = self._dup_features_df.copy()
            return

        if mode == "train":
            indices = np.asarray(self._training_indices)
        elif mode == "test":
            indices = ~np.asarray(self._training_indices)

        self._features_df = self._dup_features_df[indices].copy()

    def _retrieve(self):
        pass

    def sample_positive(self):

    def sample_negative(self):

    def sample_background(self):

class ChromatinFeaturesSampler(Sampler):

    def __init__(self, genome, genomic_features, query_features,
                 chrs_test, validation_prop=0.2,
                 radius=100, window_size=1001,
                 random_seed=436, mode="all"):
        """The positive examples we focus on in ChromatinFeaturesSampler are
        those detected in ChIP-seq, DNase-seq, and ATAC-seq experiments.
        They assay DNA binding and accessibility.

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
        genome : Genome
        query_features : GenomicFeatures
        radius : int
        padding : int
            Should be identical on both sides
        mode : {"all", "train", "test"}

        Raises
        ------
        ValueError
            - If the input str to `mode` is not one of the specified choices.
            - If the input `window_size` is less than the computed bin size.
            - If the input `window_size` is an even number.
        """
        if mode not in self.MODES:
            raise ValueError(
                "Mode must be one of {0}. Input was '{1}'.".format(
                    self.MODES, mode))

        if window_size < (1 + 2 * radius):
            raise ValueError(
                "Window size of {0} is not greater than bin "
                "size of 1 + 2 x radius {1} = {2}".format(
                    window_size, radius, 1 + 2 * radius))


        if window_size % 2 == 0:
            raise ValueError(
                "Window size must be an odd number. Input was {0}".format(
                    window_size))

        self.genome = Genome(genome)

        # used during the positive sampling step - get a random index from the
        # .bed file and the corresponding (chr, start, end, strand).
        self._features_df = pd.read_table(
            genomic_features, header=None, names=self.EXPECTED_BED_COLS,
            usecols=self.USE_BED_COLS)
        # stores a copy of the .bed file that can be used to reset
        # `self._features_df` depending on what mode is specified.
        self._dup_features_df = pd.read_table(
            genomic_features, header=None, names=self.EXPECTED_BED_COLS,
            usecols=self.USE_BED_COLS)

        self._test_indices = self._features_df["chr"].isin(test_chrs)
        self._training_indices = ~self._features_df["chr"].isin(test_chrs)
        self._validation_indices = np.random.choice(
            self._training_indices,
            size=len(self._training_indices) * validation_prop,
            replace=False)
        validation_set = set(self._validation_indices)
        self._training_indices = [ix for ix in self._training_indices
                                  if ix not in validation_set]

        features = self._features_df["feature"].unique()
        self.n_features = len(features)
        self.query_features = GenomicFeatures(query_features, features)

        # bin size = self.radius + 1 + self.radius
        self.radius = radius
        # the amount of padding is based on the window size and bin size.
        # we use the padding to incorporate sequence context around the
        # bin for which we have feature information.
        self.padding = 0

        remaining_space = window_size - self.radius * 2 - 1
        if remaining_space > 0:
            self.padding = int(remaining_space / 2)

        self.set_mode(mode)

        np.random.seed(random_seed)

        # used during the background sampling step - get a random chromosome
        # in the genome FASTA file and randomly select a position in the
        # sequence from there.
        self._randcache = []

    def set_mode(self, mode):
        """Determines what positive examples are available to sample depending
        on the mode.

        Parameters
        ----------
        mode : {"all", "train", "test}
            - all:   Use all examples in the genomic features dataset.
            - train: Use all examples except those in the holdout
                     chromosome set.
            - test:  Use only the examples in the holdout chromosome set.
        """
        if mode == "all":
            self._features_df = self._dup_features_df.copy()
            return

        if mode == "train":
            indices = np.asarray(self._training_indices)
        elif mode == "test":
            indices = ~np.asarray(self._training_indices)

        self._features_df = self._dup_features_df[indices].copy()

    def _retrieve(self, chrom, position, strand,
                  is_positive=False,
                  verbose=False):
        """
        Parameters
        ----------
        chrom : str
            e.g. "chr1".
        position : int
        strand : {'+', '-'}
        is_positive : bool, optional
            Default is True.
        verbose : bool, optional
            Default is False.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            If not `is_positive`, returns the sequence encoding and a numpy
            array of zeros (no feature labels present).
            Otherwise, returns both the sequence encoding and the feature labels
            for the specified range.
        """
        if verbose:
            print("{0}, {1}, {2}".format(chrom, position, strand))
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
                         self.query_features.n_features)))
        else:
            retrieved_data = self.query_features.get_feature_data(
                chrom, bin_start, bin_end, strand)
            return (retrieved_sequence, retrieved_data)

    def sample_background(self, verbose=False):
        """Sample a background (i.e. negative) example from the genome.

        Parameters
        ----------
        verbose : bool, optional
            Default is False.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns the sequence encoding and a numpy array of zeros (no
            feature labels present).
        """
        if len(self._randcache) == 0:
            self._randcache = list(
                np.random.choice(self.genome.chrs, size=2000))
        randchr = self._randcache.pop()
        randpos = np.random.choice(range(
            self.radius + self.padding,
            self.genome.get_chr_len(randchr) - self.radius - self.padding - 1))
        randstrand = np.random.choice(self.STRAND_SIDES)
        is_positive = self.query_features.is_positive(
            randchr, randpos - self.radius, randpos + self.radius + 1)
        if is_positive:
            print("sampled background overlapped with positive examples")
            return self.sample_background(verbose)
        else:
            print("BG: {0}, {1}, {2}".format(randchr, randpos, randstrand))
            return self._retrieve(randchr, randpos, randstrand,
                is_positive=False, verbose=verbose)

    def sample_positive(self, verbose=False):
        """Sample a positive example from the genome.

        Parameters
        ----------
        verbose : bool, optional
            Default is False.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns both the sequence encoding and the feature labels
            for the specified range.
        """
        print(self._features_df.shape[0])
        randind = np.random.randint(0, self._features_df.shape[0])
        row = self._features_df.iloc[randind]

        gene_length = row["end"] - row["start"]
        chrom = row["chr"]

        rand_in_gene = np.random.uniform() * gene_length
        position = int(
            row["start"] + rand_in_gene)

        strand = row["strand"]
        if strand == '.':
            strand = np.random.choice(self.STRAND_SIDES)

        if verbose:
            print(chrom, position, strand)
        print("PT: {0}, {1}, {2}".format(chrom, position, strand))
        seq, feats = self._retrieve(chrom, position, strand,
            is_positive=True, verbose=verbose)
        n, k = seq.shape
        if n == 0:
            print("no sequence...{0}".format(seq.shape))
            return self.sample_positive(verbose=verbose)
        else:
            return (seq, feats)

    def sample_mixture(self, positive_proportion=0.50, verbose=False):
        """Gets a mixture of positive and background samples

        Parameters
        ----------
        positive_proportion : [0.0, 1.0], float, optional
            Default is 0.50. Specify the proportion of positive examples to sample.
        verbose : bool, optional
            Default is False.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Returns both the sequence encoding and the feature labels
            for the specified range.
        """
        if np.random.uniform() < positive_proportion:
            return self.sample_positive(verbose=verbose)
        else:
            return self.sample_background(verbose=verbose)
