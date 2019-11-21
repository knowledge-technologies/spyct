import numpy as np
import scipy.sparse as sp
from spyct.node import Node
from spyct.split import learn_split


def _impurity(values):
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
                 epochs=100,
                 lr=0.01,
                 to_dense_at=1e5,
                 weight_regularization=0,
                 adam_params=(0.9, 0.999, 1e-8),
                 early_stopping_params=(3, 1e-2)):
        """
        Class for building sPyCTs and ensembles thereof.
        :param num_trees: int, (default=10).
            The number of trees in the model.
        :param bootstrapping: boolean, (default=True)
            Whether to use bootstrapped samples of the learning set to train each tree. Set to False if learning a
            single tree (if num_trees=1).
        :param max_depth: int, (default=inf)
            The maximum depth the trees can grow to. Unlimited by default.
        :param minimum_examples_to_split: int, (default=2)
            Minimum number of examples required to split an internal node. When the number of examples falls below this
            threshold, a leaf is made.
        :param epochs: int, (default=100)
            Maximum number of epochs a split is optimized for, if early stopping does not terminate the optimization
            beforehand.
        :param lr: float, (default=0.01)
            Learning rate used to optimize the splits.
        :param to_dense_at: int, (default=1e5)
            When the size of the data (#rows x #columns) falls under this threshold, the data is transformed into dense
            representation (if it was sparse before that). When there are few rows remaining, the overhead of sparse
            operations outweighs the benefits, so this helps speed up learning lower in the tree.
        :param weight_regularization: float, (default=0)
            The L1 weight regularization coefficient. Set to >0, if weight regularization is required.
        :param adam_params: tuple(b1: float, b2: float, eps: float), (default=(0.9, 0.999, 1e-8))
            Other parameters of the Adam optimizer. See [adam reference] for details.
        :param early_stopping_params: tuple(patience: int, margin: float), (default=(3, 1e-2))
            Parameters used for early stopping. After each optimization step, we check if the criterion has improved by
            at least the margin parameter specified here. If after several optimization steps (the patience parameter)
            the required margin of improvement was not achieved, the optimization stops.
        """
        self.num_trees = num_trees
        self.bootstrapping = bootstrapping
        self.max_depth = max_depth
        self.minimum_examples_to_split = minimum_examples_to_split
        self.epochs = epochs
        self.lr = lr
        self.adam_params = adam_params
        self.weight_regularization = weight_regularization
        self.early_stopping_params = early_stopping_params
        self.trees = None
        self.num_targets = 0
        self.num_nodes = 0
        self.to_dense_at = to_dense_at

    def fit(self, descriptive_data, target_data, clustering_data=None):
        """
        Build the sPyCT model from the specified data.
        :param descriptive_data: array-like or sparse matrix, shape = [n_samples, n_features]
            The features of the training examples. This data will be used for splitting the examples.
        :param target_data: array-like or sparse matrix, shape = [n_samples, n_outputs]
            The target variables of the training examples. This is what the model will predict.
        :param clustering_data: array-like or sparse matrix, shape = [n_samples, n_clustering_variables], optional
            The data used to evaluate the splits. By default it is the same as target_data (i.e., we optimize the splits
            according to the variables we wish to predict), but we can cluster the examples according to arbitrary
            variables [reference predictive clustering, or something]
        :return: None
        """

        if clustering_data is None:
            clustering_data = target_data

        # Add a column of ones for bias calculation
        if sp.isspmatrix(descriptive_data):
            descriptive_data = sp.hstack((descriptive_data, np.ones((descriptive_data.shape[0], 1)))).tocsr()
        else:
            descriptive_data = np.hstack((descriptive_data, np.ones((descriptive_data.shape[0], 1))))

        total_variance = _impurity(clustering_data)
        self.trees = []
        self.num_targets = target_data.shape[1]
        for _ in range(self.num_trees):
            if self.bootstrapping:
                rows = np.random.randint(target_data.shape[0], size=(target_data.shape[0],))
            else:
                rows = np.arange(target_data.shape[0])

            tree, num_nodes = self._grow_tree(descriptive_data[rows], target_data[rows],
                                              clustering_data[rows], total_variance)
            self.num_nodes += num_nodes
            self.trees.append(tree)

    def predict(self, descriptive_data):
        """
        Make predictions for the provided data.
        :param descriptive_data: array-like or sparse matrix of shape = [n_samples, n_features]
            The features of the examples for which predictions will be made.
        :return: array of shape = [n_samples, n_outputs]
            The predictions made by the model.
        """

        # add the bias column
        n = descriptive_data.shape[0]
        if sp.isspmatrix(descriptive_data):
            descriptive_data = sp.hstack((descriptive_data, np.ones((n, 1)))).tocsr()
        else:
            descriptive_data = np.hstack((descriptive_data, np.ones((n, 1))))

        predictions = np.zeros((n, self.num_targets))
        for tree in self.trees:
            raw_predictions = [tree.predict(descriptive_data[i]) for i in range(n)]
            predictions += np.array(raw_predictions).reshape(n, -1)
        return predictions / self.num_trees

    def _grow_tree(self, descriptive_data, target_data, clustering_data, total_variance):

        root_node = Node()
        splitting_queue = [(root_node, descriptive_data, clustering_data, target_data, total_variance)]
        num_nodes = 0
        while splitting_queue:
            node, descriptive_data, clustering_data, target_data, total_variance = splitting_queue.pop()

            # if the matrices are small, transform them into dense format
            if sp.isspmatrix(descriptive_data) and \
                    descriptive_data.shape[0] * descriptive_data.shape[1] < self.to_dense_at:
                descriptive_data = descriptive_data.toarray()

            if sp.isspmatrix(clustering_data) and \
                    clustering_data.shape[0] * clustering_data.shape[1] < self.to_dense_at:
                clustering_data = clustering_data.toarray()

            num_nodes += 1
            successful_split = False
            if total_variance > 0 and \
               node.depth < self.max_depth and \
               target_data.shape[0] >= self.minimum_examples_to_split:

                # Try to split the node
                split_weights = learn_split(descriptive_data, clustering_data, epochs=self.epochs, lr=self.lr,
                                            regularization=self.weight_regularization,
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
                    var_right = _impurity(clustering_data_right)
                    var_left = _impurity(clustering_data_left)
                    if var_right < total_variance or var_left < total_variance:
                        # We have a useful split!
                        node.split_weights = split_weights
                        node.left = Node()
                        node.right = Node()
                        splitting_queue.append((node.left, descriptive_data_left, clustering_data_left,
                                                target_data_left, var_left))
                        splitting_queue.append((node.right, descriptive_data_right, clustering_data_right,
                                                target_data_right, var_right))
                        successful_split = True

            if not successful_split:
                # Turn the node into a leaf
                node.prototype = target_data.mean(axis=0)

        return root_node, num_nodes
