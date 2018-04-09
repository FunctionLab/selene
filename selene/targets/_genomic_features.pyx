import numpy as np

cimport cython
cimport numpy as np

ctypedef np.int_t DTYPE_t
ctypedef np.float32_t FDTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False) 
def _fast_get_feature_data(int query_start, int query_end,
                           np.ndarray[FDTYPE_t, ndim=1] thresholds,
                           dict feature_index_map, 
                           rows):
    cdef int n_features = len(feature_index_map)
    cdef int query_length = query_end - query_start
    cdef int feat_start, feat_end, index_start, index_end, index_feat
    cdef np.ndarray[DTYPE_t, ndim=2] encoding = np.zeros(
        (query_length, n_features), dtype=np.int)
    cdef np.ndarray[DTYPE_t, ndim=1] targets = np.zeros(
        n_features, dtype=np.int)
    cdef list row

    if rows is None:
        return np.zeros((n_features,))
    
    for row in rows:
        feat_start = int(row[1])
        feat_end = int(row[2])
        index_start = max(0, feat_start - query_start)
        index_end = min(feat_end - query_start, query_length)
        index_feat = feature_index_map[row[3]]
        if index_start == index_end:
            index_end += 1
        encoding[index_start:index_end, index_feat] = 1

    thresholds = (thresholds * query_length - 1).clip(min=0)
    targets = (np.sum(encoding, axis=0) > thresholds.astype(int)).astype(int)
    return targets

