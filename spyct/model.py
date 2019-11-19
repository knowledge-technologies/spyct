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


class Model:

    def __init__(self,
                 num_trees=10,
                 bootstrapping=True,
                 max_depth=np.inf,
                 minimum_examples_to_split=2,
                 epochs=10,
                 lr=0.01,
                 to_dense_at=1e5,
                 adam_params=(0.9, 0.999, 1e-8),
                 early_stopping_params=(3, 1e-2)):
        self.num_trees = num_trees
        self.bootstrapping = bootstrapping
        self.max_depth = max_depth
        self.minimum_examples_to_split = minimum_examples_to_split
        self.epochs = epochs
        self.lr = lr
        self.adam_params = adam_params
        self.early_stopping_params = early_stopping_params
        self.trees = None
        self.num_targets = 0
        self.num_nodes = 0
        self.to_dense_at = to_dense_at

    def fit(self, descriptive_data, target_data, clustering_data=None):

        if clustering_data is None:
            clustering_data = target_data

        # Add a column of ones for bias calculation
        if sp.isspmatrix(descriptive_data):
            descriptive_data = sp.hstack((descriptive_data, np.ones((descriptive_data.shape[0], 1)))).tocsr()
        else:
            descriptive_data = np.hstack((descriptive_data, np.ones((descriptive_data.shape[0], 1))))

        total_variance = impurity(clustering_data)
        self.trees = []
        self.num_targets = target_data.shape[1]
        for _ in range(self.num_trees):
            if self.bootstrapping:
                rows = np.random.randint(target_data.shape[0], size=(target_data.shape[0],))
            else:
                rows = np.arange(target_data.shape[0])

            tree, num_nodes = self.grow_tree(descriptive_data[rows], target_data[rows], clustering_data[rows], total_variance)
            self.num_nodes += num_nodes
            self.trees.append(tree)

    def predict(self, data):

        # add the bias column
        if sp.isspmatrix(data):
            data = sp.hstack((data, np.ones((data.shape[0], 1)))).tocsr()
        else:
            data = np.hstack((data, np.ones((data.shape[0], 1))))

        predictions = np.zeros((data.shape[0], self.num_targets))
        n = data.shape[0]
        for tree in self.trees:
            raw_predictions = [tree.predict(data[i]) for i in range(n)]
            predictions += np.array(raw_predictions).reshape(n, -1)
        return predictions / self.num_trees

    def grow_tree(self, descriptive_data, target_data, clustering_data, total_variance):

        root_node = Node(depth=0)
        splitting_queue = [(root_node, descriptive_data, clustering_data, target_data, total_variance)]
        order = 0
        while splitting_queue:
            node, descriptive_data, clustering_data, target_data, total_variance = splitting_queue.pop()

            # if the matrices are small, transform them into dense format
            if sp.isspmatrix(descriptive_data) and descriptive_data.shape[0] * descriptive_data.shape[1] < self.to_dense_at:
                descriptive_data = descriptive_data.toarray()

            if sp.isspmatrix(clustering_data) and clustering_data.shape[0] * clustering_data.shape[1] < self.to_dense_at:
                clustering_data = clustering_data.toarray()

            node.order = order
            order += 1
            successful_split = False
            if total_variance > 0 and \
               node.depth < self.max_depth and \
               target_data.shape[0] >= self.minimum_examples_to_split:

                # Try to split the node
                split_weights = learn_split(descriptive_data, clustering_data, epochs=self.epochs, lr=self.lr,
                                            adam_params=self.adam_params,
                                            early_stopping_params=self.early_stopping_params)

                split = descriptive_data.dot(split_weights)
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
                        node.left = Node(depth=node.depth+1)
                        node.right = Node(depth=node.depth+1)
                        splitting_queue.append((node.left, descriptive_data_left, clustering_data_left, target_data_left, var_left))
                        splitting_queue.append((node.right, descriptive_data_right, clustering_data_right, target_data_right, var_right))
                        successful_split = True

            if not successful_split:
                # Turn the node into a leaf
                node.prototype = target_data.mean(axis=0)

        return root_node, order
