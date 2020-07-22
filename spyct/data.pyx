import numpy as np
import scipy.sparse as sp


cpdef Matrix matrix_from_np_or_sp(np_or_sp):
    if sp.issparse(np_or_sp):
        return csr_to_SMatrix(np_or_sp)
    else:
        return ndarray_to_DMatrix(np_or_sp)


cpdef bint missing_values_in_matrix(np_or_sp):
    if sp.issparse(np_or_sp):
        return np.isnan(np_or_sp.data).any()
    else:
        return np.isnan(np_or_sp).any()


def data_from_np_or_sp(descriptive_data, target_data,
                       clustering_data):
    cdef Data data = Data()
    data.n = descriptive_data.shape[0]
    data.d = descriptive_data.shape[1]
    data.c = clustering_data.shape[1]
    data.t = target_data.shape[1]
    data.descriptive_data = matrix_from_np_or_sp(descriptive_data)
    data.clustering_data = matrix_from_np_or_sp(clustering_data)
    data.target_data = matrix_from_np_or_sp(target_data)
    data.missing_descriptive = missing_values_in_matrix(descriptive_data)
    data.missing_clustering = missing_values_in_matrix(clustering_data)
    data.missing_target = missing_values_in_matrix(target_data)
    return data


cpdef DTYPE relative_impurity(Data data1, Data data2, DTYPE[::1] clustering_weights):
    cdef index i
    cdef DTYPE total = 0, denom = 0, w
    for i in range(data1.c):
        if clustering_weights is None:
            total += data1.impurities[i] / data2.impurities[i]
            denom += 1
        else:
            w = clustering_weights[i]
            total += w * data1.impurities[i] / data2.impurities[i]
            denom += w

    return total / denom


cdef class Data:

    def __init__(self):
        self.n = 0
        self.d = 0
        self.c = 0
        self.t = 0
        self.descriptive_data = None
        self.clustering_data = None
        self.target_data = None
        self.impurities = None
        self.missing_descriptive = False
        self.missing_clustering = False
        self.missing_target = False

    cpdef Data take_rows(self, index[::1] rows):
        cdef Data new_data = Data()
        new_data.n = rows.shape[0]
        new_data.d = self.d
        new_data.c = self.c
        new_data.t = self.t
        new_data.descriptive_data = self.descriptive_data.take_rows(rows)
        new_data.clustering_data = self.clustering_data.take_rows(rows)
        new_data.target_data = self.target_data.take_rows(rows)
        new_data.missing_descriptive = self.missing_descriptive
        new_data.missing_clustering = self.missing_clustering
        new_data.missing_target = self.missing_target
        return new_data

    cpdef void calc_impurity(self, DTYPE eps):
        self.impurities = np.empty(self.c, dtype=np.float32)
        means = np.empty(self.c, dtype=np.float32)
        if self.missing_clustering:
            self.clustering_data.column_stds_nan(eps, means, self.impurities)
        else:
            self.clustering_data.column_stds(eps, means, self.impurities)

    def split(self, Matrix split_weights, DTYPE threshold):
        cdef DTYPE[::1] scores = np.empty(self.n, dtype=np.float32)
        cdef index[::1] pos_rows, neg_rows
        cdef index pos=0, neg=0, i

        if split_weights.is_sparse:
            multiply_sparse_sparse(self.descriptive_data, split_weights, scores)
        else:
            self.descriptive_data.self_dot_vector(split_weights.data[0], scores)

        for i in range(self.n):
            if scores[i] > threshold:
                pos += 1
            else:
                neg += 1

        pos_rows = np.empty(pos, dtype=np.intp)
        neg_rows = np.empty(neg, dtype=np.intp)
        pos = 0
        neg = 0
        for i in range(self.n):
            if scores[i] > threshold:
                pos_rows[pos] = i
                pos += 1
            else:
                neg_rows[neg] = i
                neg += 1

        return self.take_rows(pos_rows), self.take_rows(neg_rows)

    cpdef index min_labelled(self):
        if self.missing_target:
            return self.target_data.min_nonnan_in_column()
        else:
            return self.n

    cpdef DTYPE total_impurity(self, DTYPE[::1] clustering_weights):
        if clustering_weights is None:
            return vector_mean(self.impurities)
        else:
            return vector_dot_vector(self.impurities, clustering_weights) / vector_sum(clustering_weights)
