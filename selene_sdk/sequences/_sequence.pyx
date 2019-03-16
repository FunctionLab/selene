# cython: language_level=3
import numpy as np

cimport cython
cimport numpy as np

ctypedef np.float32_t FDTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _fast_sequence_to_encoding(str sequence, dict base_to_index, int bases_size):
    cdef int sequence_len = len(sequence)
    cdef np.ndarray[FDTYPE_t, ndim=2] encoding = np.zeros(
        (sequence_len, bases_size), dtype=np.float32)
    cdef int index
    cdef str base
    cdef FDTYPE_t n_fill = np.divide(1, bases_size, dtype=np.float32)

    for index in range(sequence_len):
        base = sequence[index]
        if base in base_to_index:
            encoding[index, base_to_index[base]] = 1
        else:
            encoding[index, :] = n_fill
    return encoding
