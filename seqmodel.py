import math
import os

import numpy as np
import pandas as pd
from pyfaidx import Fasta
import tabix
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable


from guppy import hpy

h = hpy()

torch.set_num_threads(32)


BASES = np.array(['A', 'G', 'C', 'T'])
DIR = "/tigress/kc31/data_small"  # TODO: REMOVE


def sequence_encoding(sequence):
    """Converts an input sequence to its one hot encoding.

    Parameters
    ----------
    sequence : str
        The input sequence of length N.

    Returns
    -------
    numpy.ndarray, dtype=bool
        The N-by-4 encoding of the sequence.
    """
    encoding = np.zeros((len(sequence), 4), np.bool_)
    for base, index in zip(sequence, range(len(sequence))):
        encoding[index, :] = BASES == base
    return encoding


class Genome:

    def __init__(self, fa_file):
        """Wrapper class around the pyfaix.Fasta class.

        Parameters
        ----------
        fa_file : str
            Path to an indexed FASTA file.
            File should contain the target organism's genome sequence.

        Attributes
        ----------
        genome : Fasta
        chrs : list[str]
        """
        self.genome = Fasta(fa_file)
        self.chrs = sorted(self.genome.keys())

    def get_chr_len(self, chrom):
        """Get the length of the input chromosome.

        Parameters
        ----------
        chr : str
            e.g. "chr1".

        Returns
        -------
        int
            The length of the chromosome's genomic sequence.
        """
        return len(self.genome[chrom])

    def get_sequence(self, chrom, start, end, strand='+'):
        """Get the genomic sequence given the chromosome, sequence start,
        sequence end, and strand side.

        Parameters
        ----------
        chrom : str
            e.g. "chr1".
        start : int
        end : int
        strand : {'+', '-'}, optional
            Default is '+'.

        Returns
        -------
        str
            The genomic sequence.

        Raises
        ------
        ValueError
            If the input char to `strand` is not one of the specified choices.
        """
        if start >= len(self.genome[chrom]) or end >= len(self.genome[chrom]) or start < 0:
            print("* ~ * ~ * [EMPTY {0}, {1}, {2}] ~ * ~ *".format(chrom, start, end))
            return ""

        if strand == '+':
            return self.genome[chrom][start:end].seq
        elif strand == '-':
            return self.genome[chrom][start:end].reverse.complement.seq
        else:
            raise ValueError(
                "Strand must be one of '+' or '-'. Input was {0}".format(
                    strand))


class GenomicFeatures:

    def __init__(self, dataset, features):
        """Stores the dataset specifying sequence regions and features.
        Accepts a tabix-indexed .bed file with the following columns,
        in order:
            [chrom, start (0-based), end, strand, feature]
        Additional columns following these 5 are acceptable.

        Parameters
        ----------
        dataset : str
            Path to the tabix-indexed dataset.
        features : list[str]
            The list of features (labels) we are interested in predicting.

        Attributes
        ----------
        data : tabix.open
        n_features : int
        features_map : dict
        """
        self.data = tabix.open(dataset)
        self.n_features = len(features)

        self.features_map = dict(
            [(feat, index) for index, feat in enumerate(features)])

    def is_positive(self, chrom, start, end, threshold=0.50):
        """Determines whether the (chrom, start, end) queried
        contains features that occupy over `threshold` * 100%
        of the (start, end) region. If so, this is a positive
        example.

        Parameters
        ----------
        chrom : str
            e.g. "chr1".
        start : int
        end : int
        threshold : [0.0, 1.0], float, optional
            Default is 0.50. The threshold specifies the proportion of
            the [`start`, `end`) window that needs to be covered by
            at least one feature for the example to be considered
            positive.

        Returns
        -------
        bool
            True if this meets the criterion for a positive example,
            False otherwise.
        """
        try:
            rows = self.data.query(chrom, start, end)
            for row in rows:
                is_positive = self._is_positive_single(
                    start, end,
                    int(row[1]), int(row[2]), threshold)
                if is_positive:
                    return True
            return False
        except tabix.TabixError:
            return False

    def _is_positive_single(self, query_start, query_end,
            feat_start, feat_end, threshold):
        """Helper function to determine whether a single row from a successful
        query is considered a positive example.

        Parameters
        ----------
        query_start : int
        query_end : int
        feat_start : int
        feat_end : int
        threshold : [0.0, 1.0], float
            The threshold specifies the proportion of
            the [`start`, `end`) window that needs to be covered by
            at least one feature for the example to be considered
            positive.
        Returns
        -------
        bool
            True if this row meets the criterion for a positive example,
            False otherwise.
        """
        overlap_start = max(feat_start, query_start)
        overlap_end = min(feat_end, query_end)
        min_overlap_needed = (query_end - query_start) * threshold
        if overlap_end - overlap_start > min_overlap_needed:
            return True
        return False

    def get_feature_data(self, chrom, start, end, strand='+', threshold=0.50):
        """For a sequence of length L = `end` - `start`, return the features'
        one hot encoding corresponding to that region.
            e.g. for `n_features`, each position in that sequence will
            have a binary vector specifying whether each feature is
            present

        Parameters
        ----------
        chrom : str
            e.g. "chr1".
        start : int
        end : int
        strand : {'+', '-'}, optional
            Default is '+'.
        threshold : [0.0, 1.0], float, optional
            Default is 0.50. The threshold specifies the proportion of
            the [`start`, `end`) window that needs to be covered by
            at least one feature for the example to be considered
            positive.

        Returns
        -------
        numpy.ndarray
            shape = [L, n_features]

        Raises
        ------
        ValueError
            If the input char to `strand` is not one of the specified choices.
        """
        encoding = np.zeros((end - start, self.n_features))
        try:
            rows = self.data.query(chrom, start, end)
            if strand == '+':
                for row in rows:
                    feat_start = int(row[1])
                    feat_end = int(row[2])
                    is_positive = self._is_positive_single(
                        start, end, feat_start, feat_end, threshold)
                    if is_positive:
                        index_start = feat_start - start
                        index_end = feat_end - start
                        index_feat = self.features_map[row[4]]
                        encoding[index_start:index_end, index_feat] = 1
            elif strand == '-':
                for row in rows:
                    feat_start = int(row[1])
                    feat_end = int(row[2])
                    is_positive = self._is_positive_single(
                        start, end, feat_start, feat_end, threshold)
                    if is_positive:
                        index_start = end - feat_end
                        index_end = end - feat_start
                        index_feat = self.features_map[row[4]]
                        encoding[index_start:index_end, index_feat] = 1
            else:
                raise ValueError(
                    "Strand must be one of '+' or '-'. Input was {0}".format(
                        strand))
            return encoding
        except tabix.TabixError as e:
            print(">>>>> TABIX ERROR <<<<<")
            print(e)
            return encoding

class Sampler:

    MODES = ("all", "train", "test")
    EXPECTED_BED_COLS = (
        "chr", "start", "end", "strand", "feature") # "metadata_index")
    USE_BED_COLS = (
        "chr", "start", "end", "strand", "feature")
    STRAND_SIDES = ('+', '-')

    def __init__(self, genome, genomic_features, query_features,
                 holdout_chrs, radius=100, window_size=1001,
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
        holdout_chrs : list[str]
            Specify chromosomes to hold out (used as the test dataset).
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

        print(holdout_chrs)
        print(self._features_df["chr"].unique())
        self._training_indices = ~self._features_df["chr"].isin(holdout_chrs)

        features = self._features_df["feature"].unique()
        print(len(features))
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
        retrieved_sequence = sequence_encoding(
            self.genome.get_sequence(
                chrom, sequence_start, sequence_end, strand))
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

if __name__ == "__main__":
    n_features = 381
    hiddenSizes = [100, n_features]
    n_lstm_layers = 2
    rnn = nn.LSTM(input_size=4, hidden_size=hiddenSizes[0], num_layers=n_lstm_layers, batch_first=True, bidirectional=True)

    conv = nn.modules.container.Sequential(
        nn.Conv1d(hiddenSizes[0]*2, hiddenSizes[0]*2, 1),
        nn.ReLU(),
        nn.Conv1d(hiddenSizes[0]*2, hiddenSizes[1], 1),

        nn.Sigmoid())

    model = [rnn, conv]
    useCuda = True
    if useCuda:
        for module in model:
            module.cuda()
    padding = 400
    criterion = nn.BCELoss()
    optimizers = [optim.SGD(module.parameters(), lr=0.05, momentum=0.95) for module in model]

    sdata = Sampler(
        os.path.join(DIR, "mm10_no_alt_analysis_set_ENCODE.fasta"),
        os.path.join(DIR, "reduced_agg_beds_1.bed"),
        os.path.join(DIR, "reduced_agg_beds_1.bed.gz"),
        ["chr8", "chr9"],
        mode="train")

    def runBatch(batchSize=16, update=True, plot=False):
        window = sdata.radius * 2 + padding * 2 + 1
        inputs = np.zeros((batchSize, window, len(BASES)))
        targets = np.zeros((batchSize, sdata.radius * 2 + 1, n_features))
        for i in range(batchSize):
            sequence, target = sdata.sample_mixture()
            inputs[i, :, :] = sequence
            targets[i, :, :] = target
        if useCuda:
            inputs = Variable(torch.Tensor(inputs).cuda(), requires_grad=True)
            targets = Variable(torch.Tensor(targets).cuda())
            h0 = Variable(torch.zeros(n_lstm_layers*2, batchSize, hiddenSizes[0]).cuda())
            c0 = Variable(torch.zeros(n_lstm_layers*2, batchSize, hiddenSizes[0]).cuda())
        else:
            inputs = Variable(torch.Tensor(inputs), requires_grad=True)
            targets = Variable(torch.Tensor(targets))
            h0 = Variable(torch.zeros(n_lstm_layers * 2, batchSize, hiddenSizes[0]))
            c0 = Variable(torch.zeros(n_lstm_layers * 2, batchSize, hiddenSizes[0]))

        outputs, hn = rnn(inputs, (h0, c0))
        outputs = conv(outputs.transpose(1,2)).transpose(1,2)
        outputs = outputs[:, 400:601, :]
        #print(outputs)
        #print(outputs.size())
        #print(targets.size())
        loss = criterion(outputs,targets)

        if update:
            for module in model:
                module.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

        if plot:
            plt.figure()
            plt.plot(outputs.data.numpy().flatten(),targets.data.numpy().flatten(),'.',alpha=0.2)
            plt.show()
        return loss.data[0]


    n_epochs = 1
    for _ in range(n_epochs):
        sdata.set_mode("train")
        cumlossTrain = 0
        for _ in range(50):
            cumlossTrain = cumlossTrain + runBatch()

        sdata.set_mode("test")
        cumlossTest = 0
        for _ in range(5):
            cumlossTest = cumlossTest + runBatch(update=False)
        print("Train loss: %.5f, Test loss: %.5f." % (cumlossTrain, cumlossTest) )

    torch.save(model,os.path.join(DIR, "models/test.mm10.cpu.model"))

    print(h.heap())
