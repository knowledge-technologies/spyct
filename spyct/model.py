import numpy as np
import scipy.sparse as sp
from spyct.node import Node
from spyct.split import learn_split
from joblib import Parallel, delayed


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
                 n_jobs=1,
                 adam_beta1=0.9,
                 adam_beta2=0.999,
                 stopping_patience=3,
                 stopping_delta=1e-2,
                 eps=1e-8):
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
        :param n_jobs: int, (default=1)
            The number of parallel jobs to use when building a forest.
        :param adam_beta1: float, (default=0.9)
            Beta1 parameter for the adam optimizer. See [adam reference] for details.
        :param adam_beta2: float, (default=0.999)
            Beta2 parameter for the adam optimizer. See [adam reference] for details.
        :param eps: float, (default=1e-8)
            A tiny value added to denominators for numeric stability (Adam optimization and derivative calculation)
        :param stopping_patience: int, (default=3)
            For early stopping of optimization. If no improvement after this number of steps, the optimization is
            terminated.
        :param stopping_delta: float, (default=1e-2)
            For early stopping of optimization. The percentage increase in the value of the objective function for the
            optimization step to count as an improvement.
        """
        self.num_trees = num_trees
        self.bootstrapping = bootstrapping
        self.max_depth = max_depth
        self.minimum_examples_to_split = minimum_examples_to_split
        self.epochs = epochs
        self.lr = lr
        self.weight_regularization = weight_regularization
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.eps = eps
        self.stopping_patience = stopping_patience
        self.stopping_delta = stopping_delta
        self.to_dense_at = to_dense_at
        self.n_jobs = n_jobs

        self.trees = None                   # after fitting the model, this holds the list of trees in the ensemble
        self.sparse_target = None           # bool denoting if the matrix of target values is sparse
        self.num_targets = None             # the number of target variables
        self.num_nodes = None               # the number of nodes in the ensemble
        self.feature_importances = None     # the importance of each feature based on the learned ensemble

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

        if descriptive_data.dtype != 'f':
            descriptive_data = descriptive_data.astype('f')

        if target_data.dtype != 'f':
            target_data = target_data.astype('f')

        if clustering_data is None:
            clustering_data = target_data
        elif clustering_data.dtype != 'f':
            clustering_data = clustering_data.astype('f')

        self.sparse_target = sp.isspmatrix(target_data)

        # Add a column of ones for bias calculation
        bias_col = np.ones([descriptive_data.shape[0], 1], dtype='f')
        if sp.isspmatrix(descriptive_data):
            descriptive_data = sp.hstack((descriptive_data, bias_col)).tocsr()
        else:
            descriptive_data = np.hstack((descriptive_data, bias_col))

        total_variance = _impurity(clustering_data)
        self.num_nodes = 0
        self.num_targets = target_data.shape[1]
        self.feature_importances = np.zeros(descriptive_data.shape[1]-1)

        def tree_builder():
            if self.bootstrapping:
                rows = np.random.randint(target_data.shape[0], size=(target_data.shape[0],))
            else:
                rows = np.arange(target_data.shape[0])

            return self._grow_tree(descriptive_data[rows], target_data[rows], clustering_data[rows], total_variance)

        results = Parallel(n_jobs=self.n_jobs)(delayed(tree_builder)() for _ in range(self.num_trees))
        self.trees = []
        for tree, nodes, importances in results:
            self.trees.append(tree)
            self.num_nodes += nodes
            self.feature_importances += importances

        self.feature_importances /= self.num_trees

    def predict(self, descriptive_data):
        """
        Make predictions for the provided data.
        :param descriptive_data: array-like or sparse matrix of shape = [n_samples, n_features]
            The features of the examples for which predictions will be made.
        :return: array of shape = [n_samples, n_outputs]
            The predictions made by the model.
        """

        if descriptive_data.dtype != 'f':
            descriptive_data = descriptive_data.astype('f')

        # add the bias column
        n = descriptive_data.shape[0]
        bias_col = np.ones((n, 1), dtype='f')
        if sp.isspmatrix(descriptive_data):
            descriptive_data = sp.hstack((descriptive_data, bias_col)).tocsr()
        else:
            descriptive_data = np.hstack((descriptive_data, bias_col))

        if self.sparse_target:
            predictions = sp.csr_matrix((n, self.num_targets), dtype='f')
        else:
            predictions = np.zeros((n, self.num_targets), dtype='f')

        for tree in self.trees:
            if self.sparse_target:
                predictions += sp.vstack([tree.predict(descriptive_data[i]) for i in range(n)])
            else:
                predictions += np.vstack([tree.predict(descriptive_data[i]) for i in range(n)])
        return predictions / self.num_trees

    def get_params(self, deep=True):
        return {
            'num_trees': self.num_trees,
            'bootstrapping': self.bootstrapping,
            'max_depth': self.max_depth,
            'minimum_examples_to_split': self.minimum_examples_to_split,
            'epochs': self.epochs,
            'lr': self.lr,
            'weight_regularization': self.weight_regularization,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'stopping_patience': self.stopping_patience,
            'stopping_delta': self.stopping_delta,
            'eps': self.eps,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _grow_tree(self, descriptive_data, target_data, clustering_data, total_variance):

        root_node = Node()
        splitting_queue = [(root_node, descriptive_data, clustering_data, target_data, total_variance)]
        num_nodes = 0
        n, d = descriptive_data.shape
        feature_importance = np.zeros(d-1)
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
                                            regularization=self.weight_regularization, adam_beta1=self.adam_beta1,
                                            adam_beta2=self.adam_beta2, eps=self.eps,
                                            stopping_patience=self.stopping_patience,
                                            stopping_delta=self.stopping_delta)

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
                        feature_importance += (target_data.shape[0] / n) * \
                                              (np.abs(split_weights[:-1]) / np.linalg.norm(split_weights[:-1], ord=1))
                        node.split_weights = split_weights
                        node.left = Node(node.depth+1)
                        node.right = Node(node.depth+1)
                        splitting_queue.append((node.left, descriptive_data_left, clustering_data_left,
                                                target_data_left, var_left))
                        splitting_queue.append((node.right, descriptive_data_right, clustering_data_right,
                                                target_data_right, var_right))
                        successful_split = True

            if not successful_split:
                # Turn the node into a leaf
                if self.sparse_target:
                    node.prototype = sp.csr_matrix(target_data.mean(axis=0))
                else:
                    node.prototype = target_data.mean(axis=0)

        return root_node, num_nodes, feature_importance
