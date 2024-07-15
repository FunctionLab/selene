"""
This class contains methods to query a file of genomic coordinates,
where each row of [start, end) coordinates corresponds to a genomic target
in the sequence.

It accepts the path to a tabix-indexed .bed.gz file of genomic coordinates and
the path to an HDF5 file containing the continuous-valued targets as a matrix.

This .tsv/.bed file must contain the following columns, in order:
    chrom ('1', '2', ..., 'X', 'Y'), start (0-based), end, index
where the index is the index of the corresponding row in the HDF5 file.

Additionally, the column names should be omitted from the file itself
(i.e. there is no header and the first line in the file is the first
row of genome coordinates for a target).
"""
import types

import h5py
import numpy as np
import tabix

from functools import wraps
from .target import Target


def _get_target_data(chrom, start, end,
                      thresholds, target_index_dict, get_target_rows):
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
        An array of target thresholds, where the value in position
        `i` corresponds to the threshold for the target name that is
        mapped to index `i` by `target_index_dict`.
    target_index_dict : dict
        A dictionary mapping target names (`str`) to indices (`int`),
        where the index is the position of the target in `targets`.
    get_target_rows : types.FunctionType
        A function that takes coordinates and returns rows
        (`list(tuple(int, int, str))`).

    Returns
    -------
    numpy.ndarray, dtype=int
        A target vector where the `i`th position is equal to one if the
        `i`th target is positive, and zero otherwise.

    """
    rows = get_target_rows(chrom, start, end)
    return _fast_get_target_data(
        start, end, thresholds, target_index_dict, rows)


class GenomicFeaturesH5(Target):
    """
    Stores the dataset specifying sequence regions and targets.
    Accepts a tabix-indexed `*.bed` file with the following columns,
    in order:
    ::
        [chrom, start, end, strand, index]

    and an HDF5 file of the target values in a matrix with key `targets`.

    Note that `chrom` is interchangeable with any sort of region (e.g.
    a protein in a FAA file). Further, `start` is 0-based. The `index`
    corresponds to the row index of the targets in the HDF5 file. Lastly, any
    addition columns following the five shown above will be ignored.

    Parameters
    ----------
    tabix_path : str
        Path to the tabix-indexed dataset. Note that for the file to
        be tabix-indexed, it must have been compressed with `bgzip`.
        Thus, `input_path` should be a `*.gz` file with a
        corresponding `*.tbi` file in the same directory.
    h5_path : str
        Path to the HDF5 file of the targets matrix, with key `targets`.
    targets : list(str)
        The non-redundant list of genomic targets (i.e. labels)
        that will be predicted.
    init_unpicklable : bool, optional
        Default is False. Delays initialization until a relevant method
        is called. This enables the object to be pickled after instantiation.
        `init_unpicklable` must be `False` when multi-processing is needed e.g.
        DataLoader. Set `init_unpicklable` to True if you are using this class
        directly through Selene's API and want to access class attributes
        without having to call on a specific method in GenomicFeaturesH5.

    Attributes
    ----------
    coords : tabix.open
        The coordinates and row index stored in a tabix-indexed `*.bed` file.
    data : h5py.File
        The matrix of target data corresponding to the coordinates in `coords`.
    n_targets : int
        The number of distinct targets.
    target_index_dict : dict
        A dictionary mapping target names (`str`) to indices (`int`),
        where the index is the position of the target in `targets`.
    """

    def __init__(self,
                 tabix_path,
                 h5_path,
                 targets,
                 init_unpicklable=False):
        """
        Constructs a new `GenomicFeaturesH5` object.
        """
        self.tabix_path = tabix_path
        self.h5_path = h5_path

        self.n_targets = len(targets)

        self.target_index_dict = dict(
            [(feat, index) for index, feat in enumerate(targets)])
        self.index_target_dict = dict(list(enumerate(targets)))

        self._initialized = False

        if init_unpicklable:
            self._unpicklable_init()

    def _unpicklable_init(self):
        if not self._initialized:
            self.coords = tabix.open(self.tabix_path)
            self.data = h5py.File(self.h5_path, 'r')['targets']
            self._initialized = True

    def init(func):
        # delay initialization to allow multiprocessing
        @wraps(func)
        def dfunc(self, *args, **kwargs):
            self._unpicklable_init()
            return func(self, *args, **kwargs)
        return dfunc

    def _query_tabix(self, chrom, start, end):
        """
        Queries a tabix-indexed `*.bed` file for targets falling into
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

        Returns
        -------
        list(list(str)) or None
            A list, wherein each sub-list corresponds to a line from the
            tabix-indexed file, and each value in a sub-list corresponds
            to a column in that row. If a `tabix.TabixError` is caught,
            we assume it was because there were no targets present in
            the query region, and return `None`.

        """
        try:
            return self.coords.query(chrom, start, end)
        except tabix.TabixError:
            return None

    @init
    def is_positive(self, chrom, start, end):
        """
        Determines whether the query the `chrom` queried contains any
        genomic targets within the :math:`[start, end)` region. If so,
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
            assume the error was the result of no targets being present
            in the queried region and return `False`.
        """
        rows = self._query_tabix(chrom, start, end)
        if rows is None:
            return False
        try:
            rows.__next__()
            return True
        except StopIteration:
            return False

    @init
    def get_feature_data(self, chrom, start, end):
        """
        Computes which targets overlap with the given region.

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
            If the tabix query finds an overlap with the input region,
            `get_feature_data` will return a target vector of size
            `self.n_targets`, retrived from the matrix stored in HDF5 at the
            specified row index. If multiple overlaps are found, returns
            the average of all the overlap target rows.

            If we catch a `tabix.TabixError`, we assume the error was
            the result of there being no targets present in the queried region
            and return a `numpy.ndarray` of NaNs.

        """
        nans = np.zeros(self.n_targets) * np.nan
        rows = self._query_tabix(chrom, start, end)
        if rows is None:
            return nans

        row_targets = []
        for r in rows:
            ix = int(r[3])
            row_targets.append(self.data[ix])

        if len(row_targets) == 0:
            return nans

        row_targets = np.vstack(row_targets)
        if len(row_targets) == 1:
            return row_targets[0]

        return np.average(row_targets, axis=0)
