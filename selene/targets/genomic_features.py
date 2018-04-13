"""This class contains methods to query a file of genomic coordinates,
where each row of [start, end) coordinates corresponds to a genomic feature
in the sequence.

It accepts the path to a tabix-indexed .bed.gz file of genomic coordinates.

This .tsv/.bed file must contain the following columns, in order:
    chrom ('1', '2', ..., 'X', 'Y'), start (0-based), end, feature
Additionally, the column names should be omitted from the file itself
(i.e. there is no header and the first line in the file is the first
row of genome coordinates for a feature).
"""
import types

import tabix
import numpy as np

from .target import Target
from ._genomic_features import _fast_get_feature_data


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
                     feat_start, feat_end, threshold):
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

class GenomicFeatures(Target):

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
            The unique list of genomic features (labels) we are interested in
            predicting.
        feature_thresholds : float|dict|types.FunctionType
            A genomic region is determined to be a positive sample if at least
            1 genomic feature peak takes up a proportion of the region greater
            than or equal to the threshold specified for that feature.
            - float : a single threshold applies to all the features
                in the dataset
            - dict : str (feature) -> float (threshold). Assign different
                thresholds to different features. If a feature's
                threshold is not specified in the dict, we assume that
                a key "default" exists in the dict that has the default
                threshold value we should assign to the feature.
            - types.FunctionType : define a function that takes as input the
                feature name and returns the feature's threshold.

        Attributes
        ----------
        data : tabix.open
        n_features : int
        feature_index_map : dict
            feature (str) -> position index (int) in `features`
        index_feature_map : dict
            position index (int) -> feature (str)
        feature_thresholds : dict
            feature (str) -> threshold (float)
        """
        self.data = tabix.open(dataset)

        self.n_features = len(features)

        self.feature_index_map = dict(
            [(feat, index) for index, feat in enumerate(features)])

        self.index_feature_map = dict(list(enumerate(features)))

        self.feature_thresholds = {}
        self._feature_thresholds_vec = np.zeros(self.n_features)
        if isinstance(feature_thresholds, float):
            for i, f in enumerate(features):
                self.feature_thresholds[f] = feature_thresholds
                self._feature_thresholds_vec[i] = feature_thresholds
        elif isinstance(feature_thresholds, dict):
            for i, f in enumerate(features):
                if f in feature_thresholds:
                    self.feature_thresholds[f] = feature_thresholds[f]
                    self._feature_thresholds_vec[i] = feature_thresholds[f]
                else:
                    self.feature_thresholds[f] = feature_thresholds["default"]
                    self._feature_thresholds_vec[i] = feature_thresholds["default"]
        elif isinstance(feature_thresholds, types.FunctionType):
            for i, f in enumerate(features):
                self.feature_thresholds[f] = feature_thresholds(f)
                self._feature_thresholds_vec[i] = feature_thresholds(f)
        self._feature_thresholds_vec = self._feature_thresholds_vec.astype(np.float32)

    def _query_tabix(self, chrom, start, end):
        try:
            return self.data.query(chrom, start, end)
        except tabix.TabixError:
            return None

    def is_positive(self, chrom, start, end):
        """Determines whether the (chrom, start, end) queried
        contains any genomic features within the [start, end) region.
        If so, the query is considered positive.

        Parameters
        ----------
        chrom : str
            e.g. '1', '2', ..., 'X', 'Y'.
        start : int
        end : int

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
            e.g. '1', '2', ..., 'X', 'Y'.
        start : int
        end : int

        Returns
        -------
        numpy.ndarray
            shape = [L, n_features].
            Note that if we catch a tabix.TabixError exception, we assume
            the error was the result of no genomic features being present
            in the queried region and return a numpy.ndarray of all 0s.
        """
        return _get_feature_data(
            chrom, start, end, self._feature_thresholds_vec,
            self.feature_index_map, self._query_tabix)
