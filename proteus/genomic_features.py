"""This class contains methods to query a file of genomic coordinates,
where each row of [start, end) coordinates corresponds to a genomic feature
in the sequence.

It accepts the path to a tabix-indexed .bed.gz file of genomic coordinates.
Such a file can be created from a tab-delimited features file (.tsv/.bed) using
the following shell script:
    ../index_coordinates_file.sh
Please consult the description provided in the shell script in order to
determine what dependencies you need to install and what modifications
you should make in order to run it on your file.

This .tsv/.bed file must contain the following columns, in order:
    chrom, start (0-based), end, strand, feature
Additionally, the column names should be omitted from the file itself
(i.e. there is no header and the first line in the file is the first
row of genome coordinates for a feature).
"""
import numpy as np
import tabix


def _any_positive_rows(rows, query_start, query_end, threshold):
    if rows is None:
        return False
    for row in rows:  # features within [start, end)
        is_positive = _is_positive_row(
            query_start, query_end, int(row[1]), int(row[2]), threshold)
        if is_positive:
            return True
    return False

def _is_positive_row(query_start, query_end,
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
    else:
        return False

def _get_feature_data(query_chrom, query_start, query_end,
                      threshold, feature_index_map, get_feature_rows):
    rows = None
    if threshold < 0.50:
        rows = get_feature_rows(query_chrom, query_start, query_end)
    else:
        position = query_start + int((query_end - query_start) / 2)
        rows = get_feature_rows(query_chrom, query_start, query_end)

    n_features = len(feature_index_map)
    if rows is None:
        return np.zeros((n_features,))
    query_length = query_end - query_start
    encoding = np.zeros((query_length, n_features))
    for row in rows:
        feat_start = int(row[1])
        feat_end = int(row[2])
        index_start = max(0, feat_start - query_start)
        index_end = min(feat_end - query_start, query_length)
        index_feat = feature_index_map[row[4]]
        encoding[index_start:index_end, index_feat] = 1
    encoding = np.sum(encoding, axis=0) / query_length
    encoding = (encoding > threshold) * 1
    return encoding


class GenomicFeatures(object):

    def __init__(self, dataset, features):
        """Stores the dataset specifying sequence regions and features.
        Accepts a tabix-indexed .bed file with the following columns,
        in order:
            [chrom, start (0-based), end, strand, feature]
        Additional columns that follow these 5 are acceptable.

        Parameters
        ----------
        dataset : str
            Path to the tabix-indexed dataset. Note that for the file to
            be tabix-indexed, we must have compressed it using bgzip.
            `dataset` should be a *.gz file that has a corresponding
            *.tbi file in the same directory.
        features : list[str]
            The list of genomic features (labels) we are interested in
            predicting.

        Attributes
        ----------
        data : tabix.open
        n_features : int
        feature_index_map : dict
            feature (key) -> position index (value) in `features`
        """
        self.data = tabix.open(dataset)

        self.n_features = len(features)
        self.feature_index_map = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.index_feature_map = dict(list(enumerate(features)))

    def _query_tabix(self, chrom, start, end):
        try:
            return self.data.query(chrom, start, end)
        except tabix.TabixError:
            return None

    def is_positive(self, chrom, start, end, threshold=0.50):
        """Determines whether the (chrom, start, end) queried
        contains features that occupy over `threshold` * 100%
        of the [start, end) region. If so, this is a positive
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
        rows = self._query_tabix(chrom, start, end)
        return _any_positive_rows(rows, start, end, threshold)

    def get_feature_data(self, chrom, start, end, threshold=0.50):
        """For a sequence of length L = `end` - `start`, return the features'
        one hot encoding corresponding to that region.
            e.g. for `n_features`, each position in that sequence will
            have a binary vector specifying whether the genomic feature's
            coordinates overlap with that position.

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
        numpy.ndarray
            shape = [L, n_features].
            Note that if we catch a tabix.TabixError exception, we assume
            the error was the result of no genomic features being present
            in the queried region and return a numpy.ndarray of all 0s.
        """
        return _get_feature_data(
            chrom, start, end, threshold,
            self.feature_index_map, self._query_tabix)
