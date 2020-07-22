from libc.math cimport exp, sqrt, abs, isnan
from scipy.linalg.cython_blas cimport sgemv, sdot, sscal, sasum
from cython cimport view
import numpy as np
import scipy.sparse as sp

cdef:
    int int1=1, int0=0
    DTYPE real1=1, real0=0, real_neg1=-1, eps=1e-8


cpdef DMatrix ndarray_to_DMatrix(object data):
    cdef DMatrix matrix = DMatrix()
    matrix.data = data.copy(order='F')
    matrix.n_rows = data.shape[0]
    matrix.n_cols = data.shape[1]
    return matrix


cpdef SMatrix csr_to_SMatrix(object csr_matrix):
    cdef index i
    cdef index[::1] indptr_view = csr_matrix.indptr.astype(np.intp)
    cdef SMatrix matrix = SMatrix()
    matrix.data = csr_matrix.data.copy()
    matrix.n_rows = csr_matrix.shape[0]
    matrix.n_cols = csr_matrix.shape[1]
    matrix.column_indices = csr_matrix.indices.astype(np.intp)
    matrix.row_starts = indptr_view
    matrix.row_lengths = create_index_vector(matrix.n_rows)
    for i in range(matrix.n_rows):
        matrix.row_lengths[i] = indptr_view[i+1] - indptr_view[i]
    return matrix


cpdef DMatrix memview_to_DMatrix(DTYPE[::1] memview, index total_features,
                                 index[::1] selected_features):
    cdef DMatrix matrix = DMatrix()
    cdef DTYPE[::1] new_data = create_real_vector(total_features)
    cdef index i
    for i in range(memview.shape[0]):
        new_data[selected_features[i]] = memview[i]

    matrix.n_rows = 1
    matrix.n_cols = total_features
    matrix.data = new_data[None, :].copy_fortran()
    return matrix


cpdef SMatrix memview_to_SMatrix(DTYPE[::1] memview, index total_features,
                                 index[::1] selected_features):
    cdef SMatrix matrix = SMatrix()
    cdef index nnz = 0, i, col
    for col in range(memview.shape[0]):
        if memview[col] != 0:
            nnz += 1

    matrix.n_rows = 1
    matrix.n_cols = total_features
    matrix.data = create_real_vector(nnz)
    matrix.column_indices = create_index_vector(nnz)
    matrix.row_starts = create_index_vector(2)
    matrix.row_lengths = create_index_vector(1)

    matrix.row_starts[0] = 0
    matrix.row_starts[1] = nnz
    matrix.row_lengths[0] = nnz
    i = 0
    for col in range(memview.shape[0]):
        if memview[col] != 0:
            matrix.data[i] = memview[col]
            matrix.column_indices[i] = selected_features[col]
            i += 1

    return matrix


cpdef void multiply_sparse_sparse(SMatrix matrix, SMatrix vector, DTYPE[::1] scores):
    cdef index row
    for row in range(matrix.n_rows):
        scores[row] = multiply_sparse_sparse_row(matrix, row, vector)


cpdef DTYPE multiply_sparse_sparse_row(SMatrix matrix, index row, SMatrix vector):
    cdef index row_start, row_end, weights_end
    cdef index data_i, weights_i, data_col, weights_col
    cdef DTYPE score

    weights_end = vector.row_lengths[0]
    data_i = matrix.row_starts[row]
    row_end = data_i + matrix.row_lengths[row]
    score = 0
    weights_i = 0
    while data_i < row_end and weights_i < weights_end:
        data_col = matrix.column_indices[data_i]
        weights_col = vector.column_indices[weights_i]
        if data_col == weights_col:
            score += matrix.data[data_i] * vector.data[weights_i]
            data_i += 1
            weights_i += 1
        elif data_col < weights_col:
            data_i += 1
        else:
            weights_i += 1

    return score



###############################################################################

cdef class DMatrix(Matrix):

    def __cinit__(self):
        self.is_sparse = False
        self.data = None
        self.n_rows = 0
        self.n_cols = 0

    def __reduce__(self):
        return (ndarray_to_DMatrix, (self.to_ndarray(),))

    def to_ndarray(self):
        return np.asarray(self.data)

    cpdef DMatrix take_rows(self, index[::1] kept_rows):
        cdef index row, col, r
        cdef DMatrix new_matrix = DMatrix()
        new_matrix.n_cols = self.n_cols
        new_matrix.n_rows = kept_rows.shape[0]
        if new_matrix.n_rows > 0:
            new_matrix.data = view.array(
                shape=(new_matrix.n_rows, new_matrix.n_cols),
                itemsize=sizeof(DTYPE), format='f', mode='fortran')
            for row in range(new_matrix.n_rows):
                r = kept_rows[row]
                for col in range(new_matrix.n_cols):
                    new_matrix.data[row, col] = self.data[r, col]

        return new_matrix

    cpdef DMatrix take_columns(self, index[::1] kept_columns):
        cdef index row, col, c
        cdef DMatrix new_matrix

        if kept_columns.shape[0] == self.n_cols:
            return self.copy()
        else:
            new_matrix = DMatrix()
            new_matrix.n_rows = self.n_rows
            new_matrix.n_cols = kept_columns.shape[0]
            new_matrix.data = view.array(
                shape=(new_matrix.n_rows, new_matrix.n_cols),
                itemsize=sizeof(DTYPE), format='f', mode='fortran')
            for col in range(new_matrix.n_cols):
                c = kept_columns[col]
                for row in range(new_matrix.n_rows):
                    new_matrix.data[row, col] = self.data[row, c]
            return new_matrix

    cpdef DMatrix copy(self):
        cdef DMatrix new_matrix = DMatrix()
        new_matrix.n_cols = self.n_cols
        new_matrix.n_rows = self.n_rows
        new_matrix.data = self.data.copy_fortran()
        return new_matrix

    cpdef DTYPE[::1] row_vector(self, index row):
        cdef DTYPE[::1] vector = create_real_vector(self.n_cols)
        cdef index col
        for col in range(self.n_cols):
            vector[col] = self.data[row, col]
        return vector

    cpdef bint equal_rows(self, index r1, index r2):
        cdef index col
        for col in range(self.n_cols):
            if self.data[r1, col] != self.data[r2, col]:
                return False
        return True

    cpdef bint missing_row(self, index r):
        cdef index col
        for col in range(self.n_cols):
            if isnan(self.data[r, col]):
                return True
        return False

    cpdef void self_dot_vector(self, DTYPE[::1] vector, DTYPE[::1] result):
        cdef int m = self.n_rows, n = self.n_cols
        sgemv(b'n', &m, &n, &real1, &self.data[0, 0], &m, &vector[0], &int1,
              &real0, &result[0], &int1)

    cpdef void vector_dot_self(self, DTYPE[::1] vector, DTYPE[::1] result):
        cdef int m = self.n_rows, n = self.n_cols
        sgemv(b't', &m, &n, &real1, &self.data[0, 0], &m, &vector[0], &int1,
              &real0, &result[0], &int1)

    cpdef void column_means(self, DTYPE[::1] result):
        cdef index row, col
        cdef DTYPE mean, n
        for col in range(self.n_cols):
            mean = 0
            n = 0
            for row in range(self.n_rows):
                n += 1
                mean += (self.data[row, col] - mean) / n
            result[col] = mean

    cpdef void column_means_nan(self, DTYPE[::1] result):
        cdef index row, col
        cdef DTYPE mean, t, n
        for col in range(self.n_cols):
            n = 0
            mean = 0
            for row in range(self.n_rows):
                t = self.data[row, col]
                if not isnan(t):
                    n += 1
                    mean += (t - mean) / n
            result[col] = mean

    cpdef void column_stds(self, DTYPE fill0, DTYPE[::1] means, DTYPE[::1] stds):
        cdef index row, col
        cdef DTYPE mean, std, delta1, delta2, n, v
        for col in range(self.n_cols):
            mean = 0
            std = 0
            n = 0
            for row in range(self.n_rows):
                n += 1
                v = self.data[row, col]
                delta1 = v - mean
                mean += delta1 / n
                delta2 = v - mean
                std += delta1 * delta2
            means[col] = mean
            std = sqrt(std/n)
            if std < eps:
                stds[col] = fill0
            else:
                stds[col] = std

    cpdef void column_stds_nan(self, DTYPE fill0, DTYPE[::1] means, DTYPE[::1] stds):
        cdef index row, col
        cdef DTYPE mean, std, delta1, delta2, n, v
        for col in range(self.n_cols):
            mean = 0
            std = 0
            n = 0
            for row in range(self.n_rows):
                v = self.data[row, col]
                if not isnan(v):
                    n += 1
                    delta1 = v - mean
                    mean += delta1 / n
                    delta2 = v - mean
                    std += delta1 * delta2
            means[col] = mean
            if n > 0:
                std = sqrt(std/n)
            if std < eps:
                stds[col] = fill0
            else:
                stds[col] = std

    cpdef void impute_missing(self, DTYPE fill):
        cdef index row, col
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if isnan(self.data[row, col]):
                    self.data[row, col] = fill

    cpdef DMatrix nonmissing_matrix(self):
        cdef DMatrix new_matrix = DMatrix()
        new_matrix.n_rows = self.n_rows
        new_matrix.n_cols = self.n_cols
        new_matrix.data = view.array(shape=(new_matrix.n_rows, new_matrix.n_cols),
                                     itemsize=sizeof(DTYPE), format='f', mode='fortran')
        cdef index row, col
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if isnan(self.data[row, col]):
                    new_matrix.data[row, col] = 0
                else:
                    new_matrix.data[row, col] = 1
        return new_matrix

    cpdef void standardize_columns(self, DTYPE[::1] means, DTYPE[::1] stds):
        cdef index row, col
        cdef DTYPE m, s
        for col in range(self.n_cols):
            m = means[col]
            s = stds[col]
            for row in range(self.n_rows):
                self.data[row, col] = (self.data[row, col] - m) / s

    cpdef void unstandardize_columns(self, DTYPE[::1] means, DTYPE[::1] stds):
        cdef index row, col
        cdef DTYPE m, s
        for col in range(self.n_cols):
            m = means[col]
            s = stds[col]
            for row in range(self.n_rows):
                self.data[row, col] = s * self.data[row, col] + m

    cpdef index min_nonnan_in_column(self):
        cdef index current, min, row, col
        min = self.n_rows
        for col in range(self.n_cols):
            current = 0
            for row in range(self.n_rows):
                if not isnan(self.data[row, col]):
                    current += 1
            if current < min:
                min = current
        return min

    cpdef DTYPE cluster_rows_mse(self, DTYPE[::1] c0, DTYPE[::1] c1,
                                DTYPE[::1] left_or_right, DTYPE[::1] tiebraker):
        cdef index row, col
        cdef DTYPE d0, d1, v, t, entropy=0
        for row in range(self.n_rows):
            d0 = 0
            d1 = 0
            for col in range(self.n_cols):
                v = self.data[row, col]
                t = v - c0[col]
                d0 += t*t
                t = v - c1[col]
                d1 += t*t
            if d0 < d1:
                entropy += d0
                left_or_right[row] = 0
            elif d0 > d1:
                entropy += d1
                left_or_right[row] = 1
            elif tiebraker[row] < 0:
                left_or_right[row] = 0
            else:
                left_or_right[row] = 1

        return entropy


###############################################################################


cdef class SMatrix(Matrix):

    def __cinit__(self):
        self.is_sparse = True
        self.data = None
        self.n_rows = 0
        self.n_cols = 0
        self.row_starts = None
        self.row_lengths = None
        self.column_indices = None

    def __reduce__(self):
        return (csr_to_SMatrix, (self.to_csr(),))

    def to_csr(self):
        matrix = sp.csr_matrix((self.n_rows, self.n_cols), dtype=np.float32)
        matrix.data = np.asarray(self.data, dtype=np.float32)
        matrix.indices = np.asarray(self.column_indices, dtype=np.int32)
        matrix.indptr = np.asarray(self.row_starts, dtype=np.int32)
        return matrix

    cpdef SMatrix copy(self):
        cdef SMatrix new_matrix = SMatrix()
        new_matrix.row_starts = self.row_starts.copy()
        new_matrix.row_lengths = self.row_lengths.copy()
        if self.data.shape[0] == 0:
            new_matrix.data = self.data
            new_matrix.column_indices = self.column_indices
        else:
            new_matrix.data = self.data.copy()
            new_matrix.column_indices = self.column_indices.copy()
        new_matrix.n_cols = self.n_cols
        new_matrix.n_rows = self.n_rows
        return new_matrix

    cpdef SMatrix take_rows(self, index[::1] kept_rows):
        cdef SMatrix new_matrix = SMatrix()
        cdef index n_rows = kept_rows.shape[0]
        new_matrix.n_cols = self.n_cols
        new_matrix.n_rows = n_rows

        cdef index nnz = 0, row
        for row in range(n_rows):
            nnz += self.row_lengths[kept_rows[row]]

        new_matrix.data = create_real_vector(nnz)
        new_matrix.column_indices = create_index_vector(nnz)
        new_matrix.row_starts = create_index_vector(n_rows+1)
        new_matrix.row_lengths = create_index_vector(n_rows)

        cdef index r, i, j, k=0, rlen, rstart
        for i in range(n_rows):
            r = kept_rows[i]
            rlen = self.row_lengths[r]
            rstart = self.row_starts[r]
            new_matrix.row_starts[i] = k
            new_matrix.row_lengths[i] = rlen
            for j in range(rlen):
                new_matrix.data[k] = self.data[rstart+j]
                new_matrix.column_indices[k] = self.column_indices[rstart+j]
                k += 1

        new_matrix.row_starts[n_rows] = nnz
        return new_matrix

    cpdef SMatrix take_columns(self, index[::1] kept_columns):
        """Assumes column indices are ordered per row. The kept_columns array
           must also be ordered."""
        cdef SMatrix new_matrix
        cdef index row, data_i, row_i, col_i, row_start, row_len, i, old_col, new_col
        cdef index n_cols = kept_columns.shape[0]

        if n_cols == self.n_cols:
            return self.copy()
        else:
            new_matrix = SMatrix()
            new_matrix.n_rows = self.n_rows
            new_matrix.n_cols = n_cols
            new_matrix.data = create_real_vector(self.data.shape[0])
            new_matrix.column_indices = create_index_vector(self.data.shape[0])
            new_matrix.row_starts = create_index_vector(self.n_rows+1)
            new_matrix.row_lengths = create_index_vector(self.n_rows)
            data_i = 0
            for row in range(self.n_rows):
                new_matrix.row_starts[row] = data_i
                row_start = self.row_starts[row]
                row_len = self.row_lengths[row]
                row_i = 0
                col_i = 0
                while col_i < n_cols and row_i < row_len:
                    i = row_start + row_i
                    old_col = self.column_indices[i]
                    new_col = kept_columns[col_i]
                    if old_col == new_col:
                        new_matrix.data[data_i] = self.data[i]
                        new_matrix.column_indices[data_i] = col_i
                        data_i += 1
                        col_i += 1
                        row_i += 1
                    elif old_col < new_col:
                        row_i += 1
                    else:
                        col_i += 1

                new_matrix.row_lengths[row] = data_i - new_matrix.row_starts[row]

            new_matrix.row_starts[self.n_rows] = data_i
            new_matrix.data = new_matrix.data[:data_i].copy()
            new_matrix.column_indices = new_matrix.column_indices[:data_i].copy()
            return new_matrix

    cpdef void self_dot_vector(self, DTYPE[::1] vector, DTYPE[::1] result):
        cdef index row, col, row_len, i, j
        cdef DTYPE v
        for row in range(self.n_rows):
            row_len = self.row_lengths[row]
            if row_len > 0:
                v = 0
                i = self.row_starts[row]
                for j in range(row_len):
                    col = self.column_indices[i]
                    v += self.data[i] * vector[col]
                    i += 1
                result[row] = v
            else:
                result[row] = 0

    cpdef void vector_dot_self(self, DTYPE[::1] vector, DTYPE[::1] result):
        cdef index row, col, row_len, i, j
        cdef DTYPE v

        for col in range(self.n_cols):
            result[col] = 0

        for row in range(self.n_rows):
            row_len = self.row_lengths[row]
            if row_len > 0:
                v = vector[row]
                i = self.row_starts[row]
                for j in range(row_len):
                    col = self.column_indices[i]
                    result[col] += self.data[i] * v
                    i += 1

    cpdef void column_means(self, DTYPE[::1] result):
        cdef index col, row, row_len, i, j

        for col in range(self.n_cols):
             result[col] = 0

        for row in range(self.n_rows):
            row_len = self.row_lengths[row]
            if row_len > 0:
                i = self.row_starts[row]
                for j in range(row_len):
                    col = self.column_indices[i]
                    result[col] += self.data[i]
                    i += 1

        for col in range(self.n_cols):
            result[col] /= self.n_rows

    cpdef void column_stds(self, DTYPE fill0, DTYPE[::1] means, DTYPE[::1] stds):
        cdef index col, row, row_len, i, j
        cdef DTYPE v, w

        for col in range(self.n_cols):
            stds[col] = 0
            means[col] = 0

        for row in range(self.n_rows):
            row_len = self.row_lengths[row]
            if row_len > 0:
                i = self.row_starts[row]
                for j in range(row_len):
                    col = self.column_indices[i]
                    v = self.data[i]
                    stds[col] += v*v
                    means[col] += v
                    i += 1

        for col in range(self.n_cols):
            w = means[col] / self.n_rows
            means[col] = w
            v = stds[col] / self.n_rows - w*w
            if v < eps:
                stds[col] = fill0
            else:
                stds[col] = sqrt(v)

    cpdef void column_means_nan(self, DTYPE[::1] result):
        raise ValueError("Missing values in sparse data are not supported.")

    cpdef void column_stds_nan(self, DTYPE fill0, DTYPE[::1] means, DTYPE[::1] stds):
        raise ValueError("Missing values in sparse data are not supported.")

    cpdef index min_nonnan_in_column(self):
        raise ValueError("Missing values in sparse data are not supported.")

    cpdef void impute_missing(self, DTYPE fill):
        raise ValueError("Missing values in sparse data are not supported.")

    cpdef SMatrix nonmissing_matrix(self):
        raise ValueError("Missing values in sparse data are not supported.")

    cpdef void standardize_columns(self, DTYPE[::1] means, DTYPE[::1] stds):
        """To preserve sparsity, means are assumed to be 0."""
        cdef index col, row, row_len, i, j
        for row in range(self.n_rows):
            row_len = self.row_lengths[row]
            if row_len > 0:
                i = self.row_starts[row]
                for j in range(row_len):
                    col = self.column_indices[i]
                    self.data[i] = self.data[i] / stds[col]
                    i += 1

    cpdef void unstandardize_columns(self, DTYPE[::1] means, DTYPE[::1] stds):
        """To preserve sparsity, means are assumed to be 0."""
        cdef index col, row, row_len, i, j
        for row in range(self.n_rows):
            row_len = self.row_lengths[row]
            if row_len > 0:
                i = self.row_starts[row]
                for j in range(row_len):
                    col = self.column_indices[i]
                    self.data[i] = stds[col] * self.data[i]
                    i += 1

    cpdef DTYPE[::1] row_vector(self, index row):
        cdef DTYPE[::1] vector = create_real_vector(self.n_cols)
        cdef index col
        cdef index i = self.row_starts[row]
        cdef index rend = i + self.row_lengths[row]

        for col in range(self.n_cols):
            if i < rend and col == self.column_indices[i]:
                vector[col] = self.data[i]
                i += 1
            else:
                vector[col] = 0
        return vector

    cpdef bint equal_rows(self, index row1, index row2):
        cdef index len1, len2, i, rs1, rs2
        len1 = self.row_lengths[row1]
        len2 = self.row_lengths[row2]

        if len1 != len2:
            return False

        rs1 = self.row_starts[row1]
        rs2 = self.row_starts[row2]
        for i in range(len1):
            if self.column_indices[rs1] != self.column_indices[rs2]:
                return False
            if self.data[rs1] != self.data[rs2]:
                return False
            rs1 += 1
            rs2 += 1
        return True

    cpdef bint missing_row(self, index row):
        raise ValueError("Missing values in sparse data are not supported.")

    cpdef DTYPE cluster_rows_mse(self, DTYPE[::1] c0, DTYPE[::1] c1,
                                 DTYPE[::1] left_or_right, DTYPE[::1] tiebraker):
        cdef:
            index row, i, rend, col
            DTYPE d0, d1, entropy=0, v, t

        for row in range(self.n_rows):
            d0 = 0
            d1 = 0
            i = self.row_starts[row]
            rend = i + self.row_lengths[row]
            for col in range(self.n_cols):
                if i < rend and self.column_indices[i] == col:
                    v = self.data[i]
                    i += 1
                else:
                    v = 0
                t = c0[col] - v
                d0 += t*t
                t = c1[col] - v
                d1 += t*t

            if d0 < d1:
                entropy += d0
                left_or_right[row] = 0
            else:
                entropy += d1
                left_or_right[row] = 1

        return entropy


cdef class Matrix:
    cpdef Matrix copy(self):
        raise ValueError("Should be implemented in a subclass")
    cpdef Matrix take_rows(self, index[::1] new_rows):
        raise ValueError("Should be implemented in a subclass")
    cpdef Matrix take_columns(self, index[::1] new_columns):
        raise ValueError("Should be implemented in a subclass")
    cpdef void self_dot_vector(self, DTYPE[::1] vector, DTYPE[::1] result):
        raise ValueError("Should be implemented in a subclass")
    cpdef void vector_dot_self(self, DTYPE[::1] vector, DTYPE[::1] result):
        raise ValueError("Should be implemented in a subclass")
    cpdef void column_means(self, DTYPE[::1] result):
        raise ValueError("Should be implemented in a subclass")
    cpdef void column_means_nan(self, DTYPE[::1] result):
        raise ValueError("Should be implemented in a subclass")
    cpdef void column_stds(self, DTYPE fill0, DTYPE[::1] means, DTYPE[::1] stds):
        raise ValueError("Should be implemented in a subclass")
    cpdef void column_stds_nan(self, DTYPE fill0, DTYPE[::1] means, DTYPE[::1] stds):
        raise ValueError("Should be implemented in a subclass")
    cpdef void impute_missing(self, DTYPE fill):
        raise ValueError("Should be implemented in a subclass")
    cpdef Matrix nonmissing_matrix(self):
        raise ValueError("Should be implemented in a subclass")
    cpdef void standardize_columns(self, DTYPE[::1] means, DTYPE[::1] stds):
        raise ValueError("Should be implemented in a subclass")
    cpdef void unstandardize_columns(self, DTYPE[::1] means, DTYPE[::1] stds):
        raise ValueError("Should be implemented in a subclass")
    cpdef index min_nonnan_in_column(self):
        raise ValueError("Should be implemented in a subclass")
    cpdef bint equal_rows(self, index r1, index r2):
        raise ValueError("Should be implemented in a subclass")
    cpdef bint missing_row(self, index row):
        raise ValueError("Should be implemented in a subclass")
    cpdef DTYPE[::1] row_vector(self, index row):
        raise ValueError("Should be implemented in a subclass")
    cpdef DTYPE cluster_rows_mse(self, DTYPE[::1] c0, DTYPE[::1] c1,
                                 DTYPE[::1] left_or_right, DTYPE[::1] tiebraker):
        raise ValueError("Should be implemented in a subclass")

    cpdef DTYPE cluster_rows_dot(self, DTYPE[::1] c0, DTYPE[::1] c1,
                                 DTYPE[::1] left_or_right, DTYPE eps, DTYPE[::1] temp):
        cdef index i
        cdef DTYPE s0, s1, entropy=0, c0_norm, c1_norm
        cdef index m = self.n_rows, n = self.n_cols

        # calculate similarity
        self.self_dot_vector(c0, left_or_right)
        self.self_dot_vector(c1, temp)
        c0_norm = l2_norm(c0) + eps
        c1_norm = l2_norm(c1) + eps

        # determine nearest cluster and entropy
        for i in range(m):
            s0 = left_or_right[i] / c0_norm
            s1 = temp[i] / c1_norm
            if s0 <= s1:
                entropy += s1
                left_or_right[i] = 1
            else:
                entropy += s0
                left_or_right[i] = 0

        return entropy
