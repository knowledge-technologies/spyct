ctypedef float DTYPE
ctypedef Py_ssize_t index

cpdef DTYPE[::1] create_real_vector(index size)
cpdef index[::1] create_index_vector(index size)
cpdef DTYPE vector_dot_vector(DTYPE[::1] vector1, DTYPE[::1] vector2) nogil
cpdef void component_sum(DTYPE[::1] v1, DTYPE[::1] v2, DTYPE[::1] result) nogil
cpdef void component_diff(DTYPE[::1] v1, DTYPE[::1] v2, DTYPE[::1] result) nogil
cpdef void component_prod(DTYPE[::1] v1, DTYPE[::1] v2, DTYPE[::1] result) nogil
cpdef void component_div(DTYPE[::1] v1, DTYPE[::1] v2, DTYPE[::1] result) nogil
cpdef void reset_vector(DTYPE[::1] vector, DTYPE value) nogil
cpdef DTYPE vector_sum(DTYPE[::1] vector) nogil
cpdef DTYPE vector_mean(DTYPE[::1] vector) nogil
cpdef void vector_scalar_prod(DTYPE[::1] vector, DTYPE scalar) nogil
cpdef void vector_scalar_sum(DTYPE[::1] vector, DTYPE scalar) nogil
cpdef void impute_missing(DTYPE[::1,:] matrix, DTYPE value) nogil
cpdef DTYPE l1_norm(DTYPE[::1] vector) nogil
cpdef DTYPE l2_norm(DTYPE[::1] vector) nogil
cpdef DTYPE l05_norm(DTYPE[::1] vector) nogil
cpdef void l1_normalize(DTYPE[::1] vector) nogil
cpdef void l2_normalize(DTYPE[::1] vector) nogil
cpdef void fuzzy_split_hinge(DTYPE[::1] t, DTYPE[::1] left_selection, DTYPE[::1] right_selection, DTYPE[::1] selection_derivative) nogil
cpdef void fuzzy_split_sigmoid(DTYPE[::1] t, DTYPE[::1] left_selection, DTYPE[::1] right_selection, DTYPE[::1] selection_derivative) nogil
