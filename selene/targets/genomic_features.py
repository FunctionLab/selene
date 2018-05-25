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
    """
    TODO

    Parameters
    ----------
    rows
    query_start
    query_end
    thresholds

    Returns
    -------

    """
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
    """
    TODO

    Parameters
    ----------
    query_start
    query_end
    feat_start
    feat_end
    threshold

    Returns
    -------

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
    """
    TODO

    Parameters
    ----------
    query_chrom
    query_start
    query_end
    thresholds
    feature_index_map
    get_feature_rows

    Returns
    -------

    """
    rows = get_feature_rows(query_chrom, query_start, query_end)
    return _fast_get_feature_data(
        query_start, query_end, thresholds, feature_index_map, rows)


def _define_feature_thresholds(feature_thresholds, features):
    """
    TODO

    Parameters
    ----------
    feature_thresholds : float|dict|type.FunctionType
    features

    Returns
    -------

    """
    feature_thresholds_dict = {}
    feature_thresholds_vec = np.zeros(len(features))
    if isinstance(feature_thresholds, float):
        feature_thresholds_dict = dict.fromkeys(features, feature_thresholds)
        feature_thresholds_vec += feature_thresholds
    elif isinstance(feature_thresholds, dict):
        # assign the default value to everything first
        feature_thresholds_dict = dict.fromkeys(
            features, feature_thresholds["default"])
        feature_thresholds_vec += feature_thresholds["default"]
        for i, f in enumerate(features):
            if f in feature_thresholds:
                feature_thresholds_dict[f] = feature_thresholds[f]
                feature_thresholds_vec[i] = feature_thresholds[f]
    # this branch will not be accessed if you use a config.yml file to
    # specify input parameters
    elif isinstance(feature_thresholds, types.FunctionType):
        for i, f in enumerate(features):
            threshold = feature_thresholds(f)
            feature_thresholds_dict[f] = threshold
            feature_thresholds_vec[i] = threshold
    feature_thresholds_vec = feature_thresholds_vec.astype(np.float32)
    return feature_thresholds_dict, feature_thresholds_vec


class GenomicFeatures(Target):
    """
    Stores the dataset specifying sequence regions and features.
    Accepts a tabix-indexed `*.bed` file with the following columns,
    in order:
    ::
        [chrom, start (0-based), end, strand, feature]


    Additional columns that follow these 5 will simply be ignored.

    Parameters
    ----------
    input_path : str
        Path to the tabix-indexed dataset. Note that for the file to
        be tabix-indexed, it must have been compressed with `bgzip`.
        Thus, `input_path` should be a `*.gz` file with a
        corresponding `*.tbi` file in the same directory.
    features : list(str)
        The non-redundant list of genomic features (i.e. labels)
        that will be predicted.
    feature_thresholds : float|dict|types.FunctionType
        A genomic region is determined to be a positive sample if at
        least one genomic feature peak takes up a proportion of the
        region greater than or equal to the threshold specified for
        that feature.

        * `float` - A single threshold applies to all the features\
                    in the dataset.
        * `dict` - A dictionary mapping feature names (`str`) to \
                 threshold values (`float`), which thereby assigns\
                 different thresholds to different features. If a\
                 feature's threshold is not specified in this \
                 dictionary, then we assume that a key `"default"`\
                 exists in the dictionary that has the default \
                 threshold value we should assign to the feature \
                 name that is absent from the dictionary keys.
        * `types.FunctionType` - define a function that takes as \
                                 input the feature name and returns\
                                 the feature's threshold.

    Attributes
    ----------
    data : tabix.open
        The data stored in a tabix-indexed `*.bed` file.
    n_features : int
        The number of distinct features.
    feature_index_map : dict
        A dictionary mapping feature names (`str`) to indices (`int`),
        where the index is the position of the feature in `features`.
    index_feature_map : dict
        A dictionary mapping indices (`int`) to feature names (`str`),
        where the index is the position of the feature in the input
        features.
    feature_thresholds : dict
        A dictionary mapping feature names (`str`) to thresholds
        (`float`), where the threshold is a
        # TODO(DOCUMENTATION): FINISH.

    """

    def __init__(self, input_path, features, feature_thresholds):
        """
        Constructs a new `selene.targets.GenomicFeatures` object.
        """
        self.data = tabix.open(input_path)

        self.n_features = len(features)

        self.feature_index_map = dict(
            [(feat, index) for index, feat in enumerate(features)])

        self.index_feature_map = dict(list(enumerate(features)))

        self.feature_thresholds, self._feature_thresholds_vec = \
            _define_feature_thresholds(feature_thresholds, features)

    def _query_tabix(self, chrom, start, end):
        """
        TODO

        Parameters
        ----------
        chrom
        start
        end

        Returns
        -------

        """
        try:
            return self.data.query(chrom, start, end)
        except tabix.TabixError:
            return None

    def is_positive(self, chrom, start, end):
        """
        Determines whether the query the `chrom` queried contains any
        genomic features within the :math:`[start, end)` region. If so,
        the query is considered positive.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. '1', '2', ..., 'X', 'Y').
        start : int
            The 0-based first position in the region.
        end : int
            One past the 0-based last position in the region.
        Returns
        -------
        bool
            `True` if this meets the criterion for a positive example,
            `False` otherwise.
            Note that if we catch a `tabix.TabixError` exception, we
            assume the error was the result of no features being present
            in the queried region and return `False`.
        """
        rows = self._query_tabix(chrom, start, end)
        return _any_positive_rows(rows, start, end, self.feature_thresholds)

    def get_feature_data(self, chrom, start, end):
        """For a sequence of length :math:`L = end - start`, return the
        features' one-hot encoding corresponding to that region. For
        instance, for `n_features`, each position in that sequence will
        have a binary vector specifying whether the genomic feature's
        coordinates overlap with that position. # TODO(DOCUMENTATION): CLARIFY.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. '1', '2', ..., 'X', 'Y').
        start : int
            The 0-based first position in the region.
        end : int
            One past the 0-based last position in the region.

        Returns
        -------
        numpy.ndarray
            A :math:`L \\times N` array, where :math:`L = end - start`
            and :math:`N =` `self.n_features`. Note that if we catch a
            `tabix.TabixError`, we assume the error was the result of
            there being no features present in the queried region and
            return a `numpy.ndarray` of zeros.
        """
        return _get_feature_data(
            chrom, start, end, self._feature_thresholds_vec,
            self.feature_index_map, self._query_tabix)
