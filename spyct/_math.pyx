from scipy.linalg.cython_blas cimport sgemv, sdot, sscal, sasum
from libc.math cimport exp, sqrt, fabs, isnan
import numpy as np

cdef:
    int int1 = 1
    int int0 = 0
    float float1 = 1
    float float0 = 0


cpdef DTYPE[::1] create_real_vector(index size):
    return np.zeros(size, dtype=np.float32)


cpdef index[::1] create_index_vector(index size):
    return np.zeros(size, dtype=np.intp)


cpdef DTYPE vector_dot_vector(DTYPE[::1] vector1, DTYPE[::1] vector2) nogil:
    cdef int n = vector1.shape[0]
    return sdot(&n, &vector1[0], &int1, &vector2[0], &int1)


cpdef void component_sum(DTYPE[::1] v1, DTYPE[::1] v2, DTYPE[::1] result) nogil:
    cdef index i
    for i in range(v1.shape[0]):
        result[i] = v1[i] + v2[i]


cpdef void component_diff(DTYPE[::1] v1, DTYPE[::1] v2, DTYPE[::1] result) nogil:
    cdef index i
    for i in range(v1.shape[0]):
        result[i] = v1[i] - v2[i]


cpdef void component_prod(DTYPE[::1] v1, DTYPE[::1] v2, DTYPE[::1] result) nogil:
    cdef index i
    for i in range(v1.shape[0]):
        result[i] = v1[i] * v2[i]


cpdef void component_div(DTYPE[::1] v1, DTYPE[::1] v2, DTYPE[::1] result) nogil:
    cdef index i
    for i in range(v1.shape[0]):
        result[i] = v1[i] / v2[i]


cpdef void reset_vector(DTYPE[::1] vector, DTYPE value) nogil:
    cdef index i
    for i in range(vector.shape[0]):
        vector[i] = value


cpdef DTYPE vector_sum(DTYPE[::1] vector) nogil:
    cdef index i
    cdef DTYPE s
    s = 0
    for i in range(vector.shape[0]):
        s += vector[i]
    return s


cpdef DTYPE vector_mean(DTYPE[::1] vector) nogil:
    return vector_sum(vector) / vector.shape[0]


cpdef void vector_scalar_prod(DTYPE[::1] vector, DTYPE scalar) nogil:
    cdef int n = vector.shape[0]
    sscal(&n, &scalar, &vector[0], &int1)


cpdef void vector_scalar_sum(DTYPE[::1] vector, DTYPE scalar) nogil:
    cdef index i
    for i in range(vector.shape[0]):
        vector[i] += scalar


cpdef void impute_missing(DTYPE[::1,:] matrix, DTYPE value) nogil:
    cdef index row, col
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if isnan(matrix[row, col]):
                matrix[row, col] = value


cpdef DTYPE l1_norm(DTYPE[::1] vector) nogil:
    cdef int n = vector.shape[0]
    return sasum(&n, &vector[0], &int1)


cpdef void l1_normalize(DTYPE[::1] vector) nogil:
    cdef DTYPE norm = l1_norm(vector)
    cdef index i
    for i in range(vector.shape[0]):
        vector[i] /= norm


cpdef DTYPE l2_norm(DTYPE[::1] vector) nogil:
    return sqrt(vector_dot_vector(vector, vector))


cpdef void l2_normalize(DTYPE[::1] vector) nogil:
    cdef DTYPE norm = l2_norm(vector)
    cdef index i
    for i in range(vector.shape[0]):
        vector[i] /= norm


cpdef DTYPE l05_norm(DTYPE[::1] vector) nogil:
    cdef DTYPE norm = 0
    cdef index i
    for i in range(vector.shape[0]):
        norm += sqrt(fabs(vector[i]))
    return norm*norm


cpdef void fuzzy_split_hinge(DTYPE[::1] t, DTYPE[::1] left_selection, DTYPE[::1] right_selection, DTYPE[::1] selection_derivative) nogil:
    cdef index i
    cdef index n = t.shape[0]
    cdef DTYPE v, d
    for i in range(n):
        v = t[i]
        if v > 1:
            v = 1
            d = 0
        elif v < -1:
            v = -1
            d = 0
        else:
            d = 0.5
        v = (v + 1) / 2
        right_selection[i] = v
        left_selection[i] = 1-v
        selection_derivative[i] = d


cpdef void fuzzy_split_sigmoid(DTYPE[::1] t, DTYPE[::1] left_selection, DTYPE[::1] right_selection, DTYPE[::1] selection_derivative) nogil:
    cdef index i
    cdef index n = t.shape[0]
    cdef DTYPE e, v
    for i in range(n):
        v = t[i]
        if v > 0:
            e = exp(-v)
            right_selection[i] = 1 / (1+e)
        else:
            e = exp(v)
            right_selection[i] = e / (1+e)

        left_selection[i] = 1 - right_selection[i]
        selection_derivative[i] = right_selection[i] * left_selection[i]
