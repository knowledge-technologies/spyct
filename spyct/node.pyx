import numpy as np
from spyct._math cimport *
from spyct._matrix cimport *


cdef class Node:

    cdef:
        public index left, right, depth
        public DTYPE[::1] prototype, feature_means
        public DTYPE threshold
        public Matrix split_weights


    def __init__(self, index depth):
        self.left = -1
        self.right = -1
        self.prototype = None
        self.split_weights = None
        self.depth = depth
        self.threshold = 0
        self.feature_means = None

    def __reduce__(self):
        if self.split_weights is None:
            weights = None
        elif self.split_weights.is_sparse:
            weights = self.split_weights.to_csr()
        else:
            weights = self.split_weights.to_ndarray()
        return (rebuild, (self.left, self.right, memview_to_ndarray(self.prototype), self.split_weights.is_sparse, weights,
                          self.depth, self.threshold, memview_to_ndarray(self.feature_means)))

    cdef DTYPE test(self, object data, index row, DTYPE[::1] score):
        if self.split_weights.is_sparse:
            return multiply_sparse_sparse_row(data, row, self.split_weights)
        else:
            x = data[row]
            if self.feature_means is not None:
                nans = np.isnan(x)
                x[nans] = self.feature_means[nans]
            self.split_weights.self_dot_vector(x, score)
            return score[0]

    cdef bint is_leaf(self):
        return self.left == -1


cpdef DTYPE[::1] traverse_tree(Node[::1] node_list, object data_matrix, index example_row):
    cdef Node node
    cdef DTYPE s
    cdef DTYPE[::1] score = np.empty(1, 'f')
    node = node_list[0]
    while not node.is_leaf():
        s = node.test(data_matrix, example_row, score)
        if s <= node.threshold:
            node = node_list[node.left]
        else:
            node = node_list[node.right]
    return node.prototype


def memview_to_ndarray(memview):
    if memview is None:
        return None
    else:
        return np.asarray(memview)


cpdef Node rebuild(index left, index right, DTYPE[::1] prototype, bint sparse_weights, object weights,
                   index depth, DTYPE threshold, DTYPE[::1] feature_means):
    cdef Node result = Node(depth)
    result.left = left
    result.right = right
    result.prototype = prototype
    result.threshold = threshold
    result.feature_means = feature_means
    if weights is not None:
        if sparse_weights:
            result.split_weights = csr_to_SMatrix(weights)
        else:
            result.split_weights = ndarray_to_DMatrix(weights)
    return result