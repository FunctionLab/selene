import numpy as np

cimport cython
cimport numpy as np

ctypedef np.float32_t FDTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _fast_sequence_to_encoding(str sequence, dict base_to_index):
    cdef int sequence_len = len(sequence)
    cdef np.ndarray[FDTYPE_t, ndim=2] encoding = np.zeros(
        (sequence_len, 4), dtype=np.float32)
    cdef int index
    cdef str base
    
    for index in range(sequence_len):
        base = sequence[index]
        if base in base_to_index:
            encoding[index, base_to_index[base]] = 1
        else:
            encoding[index, :] = 0.25
    return encoding
