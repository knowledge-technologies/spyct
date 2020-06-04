import numpy as np
from spyct._matrix import multiply_sparse_sparse_row


class Node:

    def __init__(self, depth=0):
        self.left = None
        self.right = None
        self.prototype = None
        self.split_weights = None
        self.depth = depth
        self.threshold = None
        self.feature_means = None
        self.score = np.empty(1, dtype='f')

    def test(self, data, row):
        if self.split_weights.is_sparse:
            s = multiply_sparse_sparse_row(data, row, self.split_weights)
        else:
            x = data[row]
            if self.feature_means is not None:
                nans = np.isnan(x)
                x[nans] = self.feature_means[nans]
            self.split_weights.self_dot_vector(x, self.score)
            s = self.score[0]
        return s

    def is_leaf(self):
        return self.left is None and self.right is None
