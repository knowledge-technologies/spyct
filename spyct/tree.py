import numpy as np
from spyct.node import Node
from spyct.split import learn_split


def impurity(values, sparse):
    if sparse:
        means = np.asarray(values.mean(axis=0))
        means_sq = np.asarray(values.multiply(values).mean(axis=0))
        return np.sum(means_sq - means*means)
    else:
        return np.sum(np.var(values, axis=0))


class Tree:

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

    def fit(self, descriptive_data, target_data, clustering_data=None, rows=None,
            sparse_descriptive=False, sparse_target=False, sparse_clustering=False):

        if clustering_data is None:
            clustering_data = target_data
            sparse_clustering = sparse_target

        if rows is None:
            rows = np.arange(descriptive_data.shape[0])

        total_variance = impurity(clustering_data, sparse_clustering)
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
                                                        subspace_size=self.subspace_size,
                                                        adam_params=self.adam_params,
                                                        sparse_descriptive=sparse_descriptive,
                                                        sparse_clustering=sparse_clustering)
                split = descriptive_data[rows].dot(split_weights) + split_bias
                rows_right = rows[split > 0]
                rows_left = rows[split <= 0]

                if rows_right.size > 0 and rows_left.size > 0:
                    var_right = impurity(clustering_data[rows_right], sparse_clustering)
                    var_left = impurity(clustering_data[rows_left], sparse_clustering)
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
                node.prototype = target_data[rows].mean(axis=0)
        self.num_nodes = order

    def predict(self, descriptive_data):
        raw_predictions = [self.root_node.predict(descriptive_data[i]) for i in range(descriptive_data.shape[0])]
        return np.array(raw_predictions).squeeze()
