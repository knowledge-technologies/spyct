from spyct._math cimport *

cdef class Matrix:
    cdef:
        readonly index n_rows, n_cols
        readonly bint is_sparse

    cpdef Matrix copy(self)
    cpdef Matrix take_rows(self, index[::1] new_rows)
    cpdef Matrix take_columns(self, index[::1] new_columns)
    cpdef void self_dot_vector(self, DTYPE[::1] vector, DTYPE[::1] result)
    cpdef void vector_dot_self(self, DTYPE[::1] vector, DTYPE[::1] result)
    cpdef void column_means(self, DTYPE[::1] result)
    cpdef void column_means_nan(self, DTYPE[::1] result)
    cpdef void column_stds(self, DTYPE fill0, DTYPE[::1] means, DTYPE[::1] stds)
    cpdef void column_stds_nan(self, DTYPE fill0, DTYPE[::1] means, DTYPE[::1] stds)
    cpdef void impute_missing(self, DTYPE fill)
    cpdef Matrix nonmissing_matrix(self)
    cpdef void standardize_columns(self, DTYPE[::1] means, DTYPE[::1] stds)
    cpdef void unstandardize_columns(self, DTYPE[::1] means, DTYPE[::1] stds)
    cpdef index min_nonnan_in_column(self)
    cpdef DTYPE[::1] row_vector(self, index row)
    cpdef bint equal_rows(self, index r1, index r2)
    cpdef bint missing_row(self, index row)
    cpdef DTYPE cluster_rows_mse(self, DTYPE[::1] c0, DTYPE[::1] c1, DTYPE[::1] left_or_right)
    cpdef DTYPE cluster_rows_mse_nan(self, DTYPE[::1] c0, DTYPE[::1] c1, DTYPE[::1] left_or_right)
    cpdef DTYPE cluster_rows_dot(self, DTYPE[::1] c0, DTYPE[::1] c1,
                                 DTYPE[::1] left_or_right, DTYPE eps, DTYPE[::1] temp)


cpdef DMatrix ndarray_to_DMatrix(object data)
cpdef SMatrix csr_to_SMatrix(object csr_matrix)
cpdef DMatrix memview_to_DMatrix(DTYPE[::1] memview, index total_features, index[::1] selected_features)
cpdef SMatrix memview_to_SMatrix(DTYPE[::1] memview, index total_features, index[::1] selected_features)
cpdef void multiply_sparse_sparse(SMatrix data, SMatrix weights, DTYPE[::1] scores)
cpdef DTYPE multiply_sparse_sparse_row(SMatrix matrix, index row, SMatrix vector)


cdef class DMatrix(Matrix):
    cdef:
        readonly DTYPE[::1, :] data


cdef class SMatrix(Matrix):
    cdef:
        readonly DTYPE[::1] data
        readonly index[::1] row_starts
        readonly index[::1] row_lengths
        readonly index[::1] column_indices
