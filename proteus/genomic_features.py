"""This class contains methods to query a file of genomic coordinates,
where each row of [start, end) coordinates corresponds to a genomic feature
in the sequence.

It accepts the path to a tabix-indexed .bed.gz file of this feature information
which can be created from a tab-delimited features file (.tsv/.bed) using
the following shell script:
    ../index_coordinates_file.sh
Please consult the description provided in the shell script in order to
determine what dependencies you need to install and what modifications
you should make in order to run it on your file.

This .tsv/.bed file must contain the following columns, in order:
    chrom, start (0-based), end, strand, feature
Additionally, the column names should be omitted from the file itself
(i.e. there is no header and the first line in the file is the first
row or genome coordinates for a feature).
"""
import numpy as np
import tabix

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
            Path to the tabix-indexed dataset. Note that for the file to
            be tabix-indexed, we must have compressed it using bgzip.
            The `dataset` file should be *.bed.gz and have an accompanying
            *.bed.gz.tbi file in the same directory.
        features : list[str]
            The list of genomic features (labels) we are interested in
            predicting.

        Attributes
        ----------
        data : tabix.open
        n_features : int
        features_map : dict
            feature (key) -> position index (value) in `features`
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
            Note that if we catch a tabix.TabixError exception, we assume
            the error was the result of no genomic features being present
            in the queried region and return False.
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
            shape = [L, n_features].
            Note that if we catch a tabix.TabixError exception, we assume
            the error was the result of no genomic features being present
            in the queried region and return a numpy.ndarray of all 0s.

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
            return encoding

