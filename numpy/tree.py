import numpy as np
from node import Node
from split import learn_split
from numba import njit


# @njit
def impurity(values):
    return np.sum(np.var(values, axis=0))


class PCT:

    def __init__(self,
                 max_depth=np.inf,
                 subspace_size=1,
                 minimum_examples_to_split=2,
                 epochs=10,
                 lr=0.01,
                 adam_params=(0.9, 0.999, 1e-8)):

        self.max_depth = max_depth
        self.subspace_size = subspace_size
        self.minimum_examples_to_split = minimum_examples_to_split
        self.epochs = epochs
        self.lr = lr
        self.adam_params = adam_params
        self.root_node = None
        self.num_nodes = 0

    def fit(self, descriptive_data, target_data, clustering_data=None, rows=None):

        if clustering_data is None:
            clustering_data = target_data

        if rows is None:
            rows = np.arange(descriptive_data.shape[0])

        total_variance = impurity(clustering_data)
        self.root_node = Node(depth=0)
        splitting_queue = [(self.root_node, rows, total_variance)]
        order = 0
        while splitting_queue:
            node, rows, total_variance = splitting_queue.pop()
            node.order = order
            order += 1
            successful_split = False
            if total_variance > 0 and node.depth < self.max_depth and rows.shape[0] >= self.minimum_examples_to_split:

                # Try to split the node
                split_weights, split_bias = learn_split(descriptive_data[rows], clustering_data[rows],
                                                        epochs=self.epochs, lr=self.lr,
                                                        subspace_size=self.subspace_size, adam_params=self.adam_params)
                split = np.matmul(descriptive_data[rows], split_weights) + 1
                rows_right = rows[split > 0]
                rows_left = rows[split <= 0]

                if rows_right.size > 0 and rows_left.size > 0:
                    var_right = impurity(clustering_data[rows_right])
                    var_left = impurity(clustering_data[rows_left])
                    if var_right < total_variance or var_left < total_variance:
                        # We have a useful split!
                        node.split_weights = split_weights
                        node.split_bias = split_bias
                        node.left = Node(depth=node.depth+1)
                        node.right = Node(depth=node.depth+1)
                        splitting_queue.append((node.left, rows_left, var_left, ))
                        splitting_queue.append((node.right, rows_right, var_right, ))
                        successful_split = True

            if not successful_split:
                # Turn the node into a leaf
                node.prototype = np.mean(target_data[rows], axis=0)
        self.num_nodes = order

    def predict(self, descriptive_data):
        raw_predictions = [self.root_node.predict(descriptive_data[i]) for i in range(descriptive_data.shape[0])]
        return np.stack(raw_predictions)
