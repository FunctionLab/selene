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
import types

import tabix
import numpy as np

from data_utils.fastloop import _fast_get_feature_data


def _any_positive_rows(rows, query_start, query_end, thresholds):
    if rows is None:
        return False
    for row in rows:  # features within [start, end)
        is_positive = _is_positive_row(
            query_start, query_end, int(row[1]), int(row[2]), thresholds[row[3]])
        if is_positive:
            return True
    return False

def _is_positive_row(query_start, query_end,
                     feat_start, feat_end,
                     threshold):
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
    min_overlap_needed = int(
        (query_end - query_start) * threshold - 1)
    if min_overlap_needed < 0:
        min_overlap_needed = 0
    if overlap_end - overlap_start > min_overlap_needed:
        return True
    else:
        return False

def _get_feature_data(query_chrom, query_start, query_end,
                      thresholds, feature_index_map, get_feature_rows):
    rows = get_feature_rows(query_chrom, query_start, query_end)
    return _fast_get_feature_data(
        query_start, query_end, thresholds, feature_index_map, rows)

class GenomicFeatures(object):

    def __init__(self, dataset, features, feature_thresholds):
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

        self.feature_thresholds = {}
        self.feature_thresholds_vec = np.zeros(self.n_features)
        if isinstance(feature_thresholds, float):
            for i, f in enumerate(features):
                self.feature_thresholds[f] = feature_thresholds
                self.feature_thresholds_vec[i] = feature_thresholds
        elif isinstance(feature_thresholds, dict):
            for i, f in enumerate(features):
                if f in feature_thresholds:
                    self.feature_thresholds[f] = feature_thresholds[f]
                    self.feature_thresholds_vec[i] = feature_thresholds[f]
                else:
                    self.feature_thresholds[f] = feature_thresholds["default"]
                    self.feature_thresholds_vec[i] = feature_thresholds["default"]
        elif isinstance(feature_thresholds, types.FunctionType):
            for i, f in enumerate(features):
                self.feature_thresholds[f] = feature_thresholds(f)
                self.feature_thresholds_vec[i] = feature_thresholds(f)
        self.feature_thresholds_vec = self.feature_thresholds_vec.astype(np.float32)
        #print(self.feature_thresholds_vec.tolist())

    def _query_tabix(self, chrom, start, end):
        try:
            return self.data.query(chrom, start, end)
        except tabix.TabixError:
            return None

    def is_positive(self, chrom, start, end):
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
        return _any_positive_rows(rows, start, end, self.feature_thresholds)

    def get_feature_data(self, chrom, start, end):
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
            chrom, start, end, self.feature_thresholds_vec,
            self.feature_index_map, self._query_tabix)
