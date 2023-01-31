"""
This class contains methods to query a file of genomic coordinates,
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

from functools import wraps
from .target import Target
from ._genomic_features import _fast_get_feature_data


def _any_positive_rows(rows, start, end, thresholds):
    """
    Searches through a set of feature annotations for positive examples
    according to a threshold specific to each feature. For each feature
    in `rows`, the overlap between the feature and the query region must
    be greater than that feature's threshold to be considered positive.

    Parameters
    ----------
    rows : list(tuple(int, int, str)) or None
        A list of tuples of the form `(start, end, feature_name)`, or
        `None`.
    start : int
        The 0-based start coordinate of the region to query.
    end : int
        One past the last coordinate of the region to query.
    thresholds : dict
        A dictionary mapping feature names (`str`) to
        thresholds (`float`), where the threshold is the minimum
        fraction of a region that must overlap with a label for it to be
        considered a positive example of that label.

    Returns
    -------
    bool
        `True` if there is at least one feature in `rows` that meets its
        feature-specific cutoff. `False` otherwise, or if `rows==None`.

    """
    if rows is None:
        return False
    for row in rows:  # features within [start, end)
        is_positive = _is_positive_row(
            start, end, int(row[1]), int(row[2]), thresholds[row[3]])
        if is_positive:
            return True
    return False


def _is_positive_row(start, end,
                     feature_start, feature_end,
                     threshold):
    """
    Determine if a feature annotation overlaps enough with a query
    region to meet a minimal overlap threshold and be considered a
    positive example of said feature.

    Parameters
    ----------
    start : int
        The 0-based start coordinate of the query region.
    end : int
        One past the last coordinate of the query region.
    feature_start : int
        The 0-based start coordinate of the feature.
    feature_end : int
        One past the last coordinate of the feature.
    threshold : float
        The minimum fraction of the query region that a feature must
        overlap with to be considered positive.

    Returns
    -------
    bool
        `True` if the feature is a positive example and meets the
        overlap threshold. Otherwise `False`.

    """
    overlap_start = max(feature_start, start)
    overlap_end = min(feature_end, end)
    min_overlap_needed = int(
        (end - start) * threshold - 1)
    if min_overlap_needed < 0:
        min_overlap_needed = 0
    if overlap_end - overlap_start > min_overlap_needed:
        return True
    else:
        return False


def _get_feature_data(chrom, start, end,
                      thresholds, feature_index_dict, get_feature_rows, strand=None):
    """
    Generates a target vector for the given query region.

    Parameters
    ----------
     chrom : str
        The name of the region (e.g. 'chr1', 'chr2', ..., 'chrX',
        'chrY') to query inside of.
    start : int
        The 0-based start coordinate of the region to query.
    end : int
        One past the last coordinate of the region to query.
    thresholds : np.ndarray, dtype=numpy.float32
        An array of feature thresholds, where the value in position
        `i` corresponds to the threshold for the feature name that is
        mapped to index `i` by `feature_index_dict`.
    feature_index_dict : dict
        A dictionary mapping feature names (`str`) to indices (`int`),
        where the index is the position of the feature in `features`.
    get_feature_rows : types.FunctionType
        A function that takes coordinates and returns rows
        (`list(tuple(int, int, str))`).
    strand : {'+', '-', '.'}, optional
            Default is '+'. The strand the sequence is located on. '.' is treated

    Returns
    -------
    numpy.ndarray, dtype=int
        A target vector where the `i`th position is equal to one if the
        `i`th feature is positive, and zero otherwise.

    """
    rows = get_feature_rows(chrom, start, end, strand=strand)
    return _fast_get_feature_data(
        start, end, thresholds, feature_index_dict, rows)


def _define_feature_thresholds(feature_thresholds, features):
    """
    Defines the minimal overlap thresholds for the various features.

    Parameters
    ----------
    feature_thresholds : float or dict or type.FunctionType
        Either a function that takes a feature name (`str`) and returns
        a threshold (`float`) or a constant `float` that will be used
        for all features.
    features : list(str)
        A list of feature names.

    Returns
    -------
    feature_thresholds_dict, feature_thresholds_vec : \
    tuple(dict, numpy.ndarray)
        A tuple. The first element, `feature_thresholds_dict`, is a
        dictionary that maps feature names (`str`) to thresholds
        (`float`). The second element, `feature_thresholds_vec`, is an
        array of the thresholds (numpy.`float32`) where the `i`th value
        corresponds to the threshold for the `i`th feature from
        the `features` input.

    """
    feature_thresholds_dict = {}
    feature_thresholds_vec = np.zeros(len(features))
    if (isinstance(feature_thresholds, float) or
            isinstance(feature_thresholds, int)) and feature_thresholds <= 1 \
            and feature_thresholds > 0:
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
        [chrom, start, end, strand, feature]


    Note that `chrom` is interchangeable with any sort of region (e.g.
    a protein in a FAA file). Further, `start` is 0-based. Lastly, any
    addition columns following the five shown above will be ignored.

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
    feature_thresholds : float or dict or types.FunctionType or None
        Default is None. A genomic region is determined to be a
        positive sample if at least one genomic feature peak takes
        up a proportion of the region greater than or equal to
        the threshold specified for that feature.

        * `None` - No thresholds specified. All features found in\
                   a query region are annotated to that region.
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

    init_unpicklable : bool, optional
        Default is False. Delays initialization until a relevant method
        is called. This enables the object to be pickled after instantiation.
        `init_unpicklable` must be `False` when multi-processing is needed e.g.
        DataLoader. Set `init_unpicklable` to True if you are using this class
        directly through Selene's API and want to access class attributes
        without having to call on a specific method in GenomicFeatures.

    Attributes
    ----------
    data : tabix.open
        The data stored in a tabix-indexed `*.bed` file.
    n_features : int
        The number of distinct features.
    feature_index_dict : dict
        A dictionary mapping feature names (`str`) to indices (`int`),
        where the index is the position of the feature in `features`.
    index_feature_dict : dict
        A dictionary mapping indices (`int`) to feature names (`str`),
        where the index is the position of the feature in the input
        features.
    feature_thresholds : dict or None

        * `dict` - A dictionary mapping feature names (`str`) to thresholds\
        (`float`), where the threshold is the minimum overlap that a\
        feature annotation must have with a query region to be\
        considered a positive example of that feature.
        * `None` - No threshold specifications. Assumes that all features\
        returned by a tabix query are annotated to the query region.

    """

    def __init__(self, input_path, features, feature_thresholds=None, init_unpicklable=False):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_path = input_path
        self.n_features = len(features)

        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])

        self.index_feature_dict = dict(list(enumerate(features)))

        if feature_thresholds is None:
            self.feature_thresholds = None
            self._feature_thresholds_vec = None
        else:
            self.feature_thresholds, self._feature_thresholds_vec = \
                _define_feature_thresholds(feature_thresholds, features)
        self._initialized = False

        if init_unpicklable:
            self._unpicklable_init()

    def _unpicklable_init(self):
        if not self._initialized:
            self.data = tabix.open(self.input_path)
            self._initialized = True

    def init(func):
        # delay initialization to allow multiprocessing
        @wraps(func)
        def dfunc(self, *args, **kwargs):
            self._unpicklable_init()
            return func(self, *args, **kwargs)
        return dfunc

    def _query_tabix(self, chrom, start, end, strand=None):
        """
        Queries a tabix-indexed `*.bed` file for features falling into
        the specified region.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. '1', '2', ..., 'X', 'Y') to
            query in.
        start : int
            The 0-based start position of the query coordinates.
        end : int
            One past the last position of the query coordinates.
        strand : {'+', '-', '.'}
            The strand the sequence is located on. '.' is treated


        Returns
        -------
        list(list(str)) or None
            A list, wherein each sub-list corresponds to a line from the
            tabix-indexed file, and each value in a sub-list corresponds
            to a column in that row. If a `tabix.TabixError` is caught,
            we assume it was because there were no features present in
            the query region, and return `None`.

        """
        try:
            tabix_query = self.data.query(chrom, start, end)
            if strand:
                return [line for line in tabix_query if str(line[4]) == strand] # strand specificity
            else: # not strand specific
                return tabix_query
        except tabix.TabixError:
            return None

    @init
    def is_positive(self, chrom, start, end, strand=None):
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
        rows = self._query_tabix(chrom, start, end, strand=strand)
        return _any_positive_rows(rows, start, end, self.feature_thresholds)

    @init
    def get_feature_data(self, chrom, start, end, strand=None):
        """
        Computes which features overlap with the given region.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. '1', '2', ..., 'X', 'Y').
        start : int
            The 0-based first position in the region.
        end : int
            One past the 0-based last position in the region.
        strand : {'+', '-', '.'}, optional
            Default is None (no strand provided).
            The strand the sequence is located on. '.' is treated

        Returns
        -------
        numpy.ndarray
            A target vector of size `self.n_features` where the `i`th
            position is equal to one if the `i`th feature is positive,
            and zero otherwise.

            NOTE: If we catch a `tabix.TabixError`, we assume the error was
            the result of there being no features present in the queried region
            and return a `numpy.ndarray` of zeros.

        """
        if self._feature_thresholds_vec is None:
            features = np.zeros(self.n_features)
            rows = self._query_tabix(chrom, start, end, strand=strand) # strand specificity
            if not rows:
                return features
            for r in rows:
                feature = r[3]
                ix = self.feature_index_dict[feature]
                features[ix] = 1
            return features
        return _get_feature_data(
            chrom, start, end, self._feature_thresholds_vec,
            self.feature_index_dict, self._query_tabix, strand=strand)