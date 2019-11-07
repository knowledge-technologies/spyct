import numpy as np
import scipy.sparse as sp
from spyct.node import Node
from spyct.split import learn_split


def impurity(values):
    if sp.isspmatrix(values):
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

    def fit(self, descriptive_data, target_data, clustering_data=None, to_dense_at=1e5):

        if clustering_data is None:
            clustering_data = target_data

        total_variance = impurity(clustering_data)
        self.root_node = Node(depth=0)
        splitting_queue = [(self.root_node, descriptive_data, clustering_data, target_data, total_variance)]
        order = 0
        while splitting_queue:
            node, descriptive_data, clustering_data, target_data, total_variance = splitting_queue.pop()

            # if the matrices are small, transform them into dense format
            if sp.isspmatrix(descriptive_data) and descriptive_data.shape[0] * descriptive_data.shape[1] < to_dense_at:
                descriptive_data = descriptive_data.toarray()

            if sp.isspmatrix(clustering_data) and clustering_data.shape[0] * clustering_data.shape[1] < to_dense_at:
                clustering_data = clustering_data.toarray()

            node.order = order
            order += 1
            successful_split = False
            if total_variance > 0 and \
               node.depth < self.max_depth and \
               target_data.shape[0] >= self.minimum_examples_to_split:

                # Try to split the node
                split_weights, split_bias = learn_split(descriptive_data, clustering_data,
                                                        epochs=self.epochs, lr=self.lr,
                                                        subspace_size=self.subspace_size,
                                                        adam_params=self.adam_params)

                split = descriptive_data.dot(split_weights) + split_bias
                descriptive_data_right = descriptive_data[split > 0]
                descriptive_data_left = descriptive_data[split <= 0]
                clustering_data_right = clustering_data[split > 0]
                clustering_data_left = clustering_data[split <= 0]
                target_data_right = target_data[split > 0]
                target_data_left = target_data[split <= 0]

                if target_data_right.shape[0] > 0 and target_data_left.shape[0] > 0:
                    var_right = impurity(clustering_data_right)
                    var_left = impurity(clustering_data_left)
                    if var_right < total_variance or var_left < total_variance:
                        # We have a useful split!
                        node.split_weights = split_weights
                        node.split_bias = split_bias
                        node.left = Node(depth=node.depth+1)
                        node.right = Node(depth=node.depth+1)
                        splitting_queue.append((node.left, descriptive_data_left, clustering_data_left, target_data_left, var_left))
                        splitting_queue.append((node.right, descriptive_data_right, clustering_data_right, target_data_right, var_right))
                        successful_split = True

            if not successful_split:
                # Turn the node into a leaf
                node.prototype = target_data.mean(axis=0)
        self.num_nodes = order

    def predict(self, descriptive_data):
        n = descriptive_data.shape[0]
        raw_predictions = [self.root_node.predict(descriptive_data[i]) for i in range(descriptive_data.shape[0])]
        return np.array(raw_predictions).reshape(n, -1)
