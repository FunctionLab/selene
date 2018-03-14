import numpy as np

cimport cython
cimport numpy as np

ctypedef np.int_t DTYPE_t
ctypedef np.float32_t FDTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _fast_sequence_to_encoding(str sequence, dict bases_encoding):
    cdef int sequence_len = len(sequence)
    cdef np.ndarray[FDTYPE_t, ndim=2] encoding = np.zeros(
        (sequence_len, 4), dtype=np.float32)
    cdef int index
    cdef str base
    
    sequence = str.upper(sequence)

    for index in range(sequence_len):
        base = sequence[index]
        if base in bases_encoding:
            encoding[index, bases_encoding[base]] = 1
        else:
            encoding[index, :] = 0.25
    return encoding

@cython.boundscheck(False)
@cython.wraparound(False) 
def _fast_get_feature_data(int query_start, int query_end, double threshold,
                      dict feature_index_map, rows):
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
    targets = (
        np.around(np.sum(encoding, axis=0) / query_length, 2) >= threshold).astype(int)
    return targets

