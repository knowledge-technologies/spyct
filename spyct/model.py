import numpy as np
import scipy.sparse as sp
from spyct.node import Node, traverse_tree
from spyct.grad_split import GradSplitter
from spyct.svm_split import SVMSplitter
from spyct.data import data_from_np_or_sp, relative_impurity
from spyct._matrix import memview_to_SMatrix, memview_to_DMatrix, csr_to_SMatrix
from joblib import Parallel, delayed

DTYPE = 'f'


class Model:

    def __init__(self,
                 splitter='grad',
                 objective='mse',
                 boosting=False,
                 boosting_step=0.1,
                 num_trees=100,
                 max_features=1.0,
                 bootstrapping=None,
                 max_depth=np.inf,
                 min_examples_to_split=2,
                 min_impurity_decrease=0.05,
                 n_jobs=1,
                 standardize_descriptive=True,
                 standardize_clustering=True,
                 max_iter=100,
                 lr=0.1,
                 C=10,
                 balance_classes=True,
                 clustering_iterations=10,
                 tol=1e-2,
                 eps=1e-8,
                 adam_beta1=0.9,
                 adam_beta2=0.999,
                 random_state=None):
        """
        Class for building spycts and ensembles thereof.

        :param splitter: string, (default="grad")
            Determines which split optimizer to use. Supported values are "grad" and "svm".
        :param objective: string, (default="mse")
            Determines the objective to optimize when splitting the data. Only "mse" is supported and tested.
        :param num_trees: int, (default=100).
            The number of trees in the model.
        :param max_features: int, float, "sqrt", "log" (default=1.0)
            The number of features to consider when optimizing the splits:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log", then `max_features=log2(n_features)`.
            At least one feature is always considered.
        :param bootstrapping: boolean, (default=None)
            Whether to use bootstrapped samples of the learning set to train each
            tree. If not set, bootstrapping is used when learning more than one tree.
        :param max_depth: int, (default=inf)
            The maximum depth the trees can grow to. Unlimited by default.
        :param min_examples_to_split: int, (default=2)
            Minimum number of examples required to split an internal node. When the number of examples falls below this
            threshold, a leaf node is made.
        :param min_impurity_decrease: float, (default=0)
            Minimum relative impurity decrease of at least one subset produced by a split. If not achieved, the
            splitting stops and a leaf node is made.
        :param n_jobs: int, (default=1)
            The number of parallel jobs to use when building a forest. Uses process based parallelism with joblib.
        :param standardize_descriptive: boolean, (default=True)
            Determines if the descriptive data is standardized to mean=0 and std=1 when learning weights for each split.
            If the data is sparse, mean is assumed to be 0, to preserve sparsity.
        :param standardize_clustering: boolean, (default=True)
            Determines if the clustering data is standardized to mean=0 and std=1 when learning weights for each split.
            If the data is sparse, mean is assumed to be 0, to preserve sparsity.
        :param max_iter: int, (default=100)
            Maximum number of iterations a split is optimized for, if early stopping does not terminate the optimization
            beforehand.
        :param lr: float, (default=0.01)
            Learning rate used to optimize the splits in the "variance" splitter.
        :param C: float, (default=0)
            Split weight regularization parameter. The strength of the regularization is inversely proportional to C.
        :param balance_classes: boolean, (default=True)
            Used by the "svm" splitter. If True, automatically adjust weights of classes when learning the split to be
            inversely proportional to their frequencies in the data.
        :param tol: float, (default=0)
            Tolerance for stopping criteria.
        :param eps: float, (default=1e-8)
            A tiny value added to denominators for numeric stability where division by zero could occur.
        :param adam_beta1: float, (default=0.9)
            Beta1 parameter for the adam optimizer. See [adam reference] for details. Used by the "grad" splitter.
        :param adam_beta2: float, (default=0.999)
            Beta2 parameter for the adam optimizer. See [adam reference] for details. Used by the "grad" splitter.
        :param random_state: RandomState instance, int, (default=None)
            If provided, the RandomState instance will be used for any RNG. If provided an int, a RandomState instance
            with the provided int as the seed will be used.
        """
        self.num_trees = num_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_examples_to_split = min_examples_to_split
        self.n_jobs = n_jobs
        self.min_impurity_decrease = min_impurity_decrease

        # universal parameters
        self.splitter = splitter
        self.objective = objective
        self.boosting = boosting
        self.boosting_step = boosting_step
        self.max_iter = max_iter
        self.standardize_descriptive = standardize_descriptive
        self.standardize_clustering = standardize_clustering
        self.tol = tol
        self.eps = eps
        # variance parameters
        self.lr = lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        # svm parameters
        self.C = C
        self.balance_classes = balance_classes
        self.clustering_iterations = clustering_iterations

        if type(random_state) is int:
            self.rng = np.random.RandomState(random_state)
        elif type(random_state) is np.random.RandomState:
            self.rng = random_state
        else:
            self.rng = np.random.RandomState()

        if bootstrapping is not None:
            self.bootstrapping = bootstrapping
        else:
            self.bootstrapping = num_trees > 1

        if splitter not in ['svm', 'grad']:
            raise ValueError('Unknown splitter specified. Supported values are "grad" and "svm".')

        if objective not in ['mse', 'dot']:
            raise ValueError('Unknown objective function specified. Possible values are "mse" and "dot".')

        self.trees = None  # after fitting the model, this holds the list of trees in the ensemble
        self.sparse_target = None  # bool denoting if the matrix of target values is sparse
        self.num_targets = None  # the number of target variables
        self.num_nodes = None  # the number of nodes in the ensemble
        self.feature_importances = None  # the importance of each feature based on the learned ensemble
        self.total_iterations = None  # The total number of optimization iterations
        self.max_relative_impurity = None  # The maximum relative impurity remained after a split

    def fit(self, descriptive_data, target_data, clustering_data=None, clustering_weights=None):
        """
        Learn the spyct model from the specified data.
        :param descriptive_data: array-like or sparse matrix, shape = [n_samples, n_features]
            The features of the training examples. This data will be used for splitting the examples.
        :param target_data: array-like or sparse matrix, shape = [n_samples, n_outputs]
            The target variables of the training examples. This is what the model will predict.
        :param clustering_data: array-like or sparse matrix, shape = [n_samples, n_clustering_variables], optional
            The data used to evaluate the splits. By default it is the same as target_data (i.e., we optimize the splits
            according to the variables we wish to predict), but we can cluster the examples according to arbitrary
            variables [reference predictive clustering, or something]
        :param clustering_weights: array-like, shape = [n_clustering_variables], optional
            Optional weights for the clustering variables, enables giving different priorities to different targets. By
            default, all targets have the same weight.
        :return: None
        """

        if len(descriptive_data.shape) != 2:
            raise ValueError("Descriptive data must have exactly 2 dimensions.")
        if clustering_data is not None and len(clustering_data.shape) != 2:
            raise ValueError("Clustering data must have exactly 2 dimensions.")
        if len(target_data.shape) != 2:
            raise ValueError("Target data must have exactly 2 dimensions.")

        # calculate the number of features to consider at each split
        if type(self.max_features) is int:
            num_features = self.max_features
        elif type(self.max_features) is float:
            num_features = self.max_features * descriptive_data.shape[1]
        elif self.max_features == 'sqrt':
            num_features = np.sqrt(descriptive_data.shape[1])
        elif self.max_features == 'log':
            num_features = np.log2(descriptive_data.shape[1])
        else:
            raise ValueError("The max_features parameter was specified incorrectly.")

        num_features = max(1, int(np.ceil(num_features)))

        # If data is sparse, make sure it has no missing values and the format is CSR.
        # Make sure the clustering weights are contiguous.
        # Make sure the numeric precision is correct.

        bias_col = np.ones([descriptive_data.shape[0], 1], dtype=DTYPE)  # column of ones for bias calculation
        if sp.issparse(descriptive_data):
            descriptive_data = descriptive_data.astype(DTYPE, copy=False)
            descriptive_data = sp.hstack((descriptive_data, bias_col)).tocsr()
        else:
            descriptive_data = descriptive_data.astype(DTYPE, order='C', copy=False)
            descriptive_data = np.hstack((descriptive_data, bias_col))

        self.sparse_target = sp.issparse(target_data)
        if self.sparse_target:
            target_data = target_data.astype(DTYPE, copy=False)
            target_data = target_data.tocsr()
        else:
            target_data = target_data.astype(DTYPE, order='C', copy=False)

        if clustering_data is None:
            clustering_data = target_data
        elif sp.issparse(clustering_data):
            clustering_data = clustering_data.astype(DTYPE, copy=False)
            clustering_data = clustering_data.tocsr()
        else:
            clustering_data = clustering_data.astype(DTYPE, order='C', copy=False)

        if clustering_weights is not None:
            clustering_weights = clustering_weights.astype(DTYPE, order='C', copy=False)# + self.eps
            clustering_weights /= clustering_weights.max()
            self.standardize_clustering = True

        # Initialize everything
        all_data = data_from_np_or_sp(descriptive_data, target_data, clustering_data)
        self.num_nodes = 0
        self.total_iterations = 0
        self.num_targets = target_data.shape[1]
        self.max_relative_impurity = 1 - self.min_impurity_decrease
        self.feature_importances = np.zeros(descriptive_data.shape[1] - 1)

        # Function that wraps tree building for parallelization. Bootstrapping if more than one tree.
        def tree_builder(seed):
            rng = np.random.RandomState(seed)
            if self.bootstrapping:
                rows = rng.randint(target_data.shape[0], size=target_data.shape[0], dtype=np.intp)
                data = all_data.take_rows(rows)
            else:
                data = all_data

            return self._grow_tree(data, clustering_weights, num_features, rng)

        # Learn the trees
        if self.boosting:
            if False and self.sparse_target:
                predictions = sp.csr_matrix((all_data.n, self.num_targets), dtype=DTYPE)
                stack = sp.vstack
            else:
                predictions = np.zeros((all_data.n, self.num_targets), dtype=DTYPE)
                stack = np.vstack
            if sp.issparse(descriptive_data):
                inputs = csr_to_SMatrix(descriptive_data)
            else:
                inputs = descriptive_data
            results = []
            for i in range(self.num_trees):
                temp = (target_data - predictions)
                temp *= self.boosting_step
                all_data = data_from_np_or_sp(descriptive_data, temp, temp)
                if self.bootstrapping:
                    rows = self.rng.randint(temp.shape[0], size=temp.shape[0], dtype=np.intp)
                    all_data = all_data.take_rows(rows)
                t = self._grow_tree(all_data, clustering_weights, num_features, self.rng)
                results.append(t)
                predictions += stack([traverse_tree(t[0], inputs, i) for i in range(all_data.n)])

        else:
            seeds = self.rng.randint(10**9, size=self.num_trees)
            if self.n_jobs > 1:
                results = Parallel(n_jobs=self.n_jobs)(delayed(tree_builder)(seeds[i]) for i in range(self.num_trees))
            else:
                results = [tree_builder(seeds[i]) for i in range(self.num_trees)]

        # Collect the results
        self.trees = []
        for node_list, importances, iterations in results:
            self.trees.append(node_list)
            self.num_nodes += len(node_list)
            self.feature_importances += importances
            self.total_iterations += iterations

        self.feature_importances /= self.num_trees
        return self

    def predict(self, descriptive_data, used_trees=None):
        """
        Make predictions for the provided data.
        :param descriptive_data: array-like or sparse matrix of shape = [n_samples, n_features]
            The features of the examples for which predictions will be made.
        :param used_trees: int, (default=None)
            Gives an option to only use a subset of trees to make predictions, useful for evaluation different ensemble
            sizes in one go. If None, all trees will be used.
        :return: array of shape = [n_samples, n_outputs]
            The predictions made by the model.
        """

        # add the bias column
        n = descriptive_data.shape[0]
        descriptive_data = descriptive_data.astype(DTYPE, copy=False)
        if sp.issparse(descriptive_data):
            descriptive_data = csr_to_SMatrix(descriptive_data.to_csr())

        if False and self.sparse_target:
            predictions = sp.csr_matrix((n, self.num_targets), dtype=DTYPE)
            stack = sp.vstack
        else:
            predictions = np.zeros((n, self.num_targets), dtype=DTYPE)
            stack = np.vstack

        def predictor(node_list, descriptive_data):
            descriptive_data = descriptive_data.copy()
            return stack([traverse_tree(node_list, descriptive_data, i) for i in range(n)])

        n_trees = self.num_trees if used_trees is None else used_trees
        if self.n_jobs > 1:
            preds = Parallel(n_jobs=self.n_jobs)(
                delayed(predictor)(node_list, descriptive_data) for node_list in self.trees[:n_trees])
            predictions = np.sum(preds, axis=0)
        else:
            for node_list in self.trees[:n_trees]:
                predictions += predictor(node_list, descriptive_data)
        if not self.boosting:
            predictions /= n_trees
        return predictions

    def get_params(self, **kwargs):
        return {
            'splitter': self.splitter,
            'objective': self.objective,
            'num_trees': self.num_trees,
            'max_features': self.max_features,
            'bootstrapping': self.bootstrapping,
            'max_depth': self.max_depth,
            'min_examples_to_split': self.min_examples_to_split,
            'min_impurity_decrease': self.min_impurity_decrease,
            'n_jobs': self.n_jobs,
            'standardize_descriptive': self.standardize_descriptive,
            'standardize_clustering': self.standardize_clustering,
            'max_iter': self.max_iter,
            'lr': self.lr,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'tol': self.tol,
            'C': self.C,
            'balance_classes': self.balance_classes,
            'clustering_iterations': self.clustering_iterations,
            'eps': self.eps,
        }

    def set_params(self, **params):

        for key, value in params.items():
            if key == 'random_state':
                if type(value) is int:
                    self.rng = np.random.RandomState(value)
                elif type(value) is np.random.RandomState:
                    self.rng = value
                else:
                    self.rng = np.random.RandomState()
            else:
                setattr(self, key, value)

        if 'bootstrapping' not in params:
            self.bootstrapping = self.num_trees > 1

        if self.splitter not in ['svm', 'grad']:
            raise ValueError('Unknown splitter specified. Supported values are "grad" and "svm".')

        if self.objective not in ['mse', 'dot']:
            raise ValueError('Unknown objective function specified. Possible values are "mse" and "dot".')

        return self

    def _grow_tree(self, root_data, clustering_weights, num_features, rng):
        """Grow a single tree."""

        if self.splitter == 'grad':
            splitter = GradSplitter(n=root_data.n, d=num_features+1, c=root_data.c,
                                    clustering_weights=clustering_weights,
                                    max_iter=self.max_iter, learning_rate=self.lr,
                                    regularization=1/self.C, adam_beta1=self.adam_beta1,
                                    adam_beta2=self.adam_beta2, eps=self.eps,
                                    tol=self.tol, standardize_descriptive=self.standardize_descriptive,
                                    standardize_clustering=self.standardize_clustering, rng=rng,
                                    objective=self.objective)
        else:
            splitter = SVMSplitter(n=root_data.n, d=num_features+1, c=root_data.c,
                                   clustering_weights=clustering_weights, opt_iter=self.max_iter,
                                   cluster_iter=self.clustering_iterations, eps=self.eps, C=self.C, tol=self.tol,
                                   balance_classes=self.balance_classes,
                                   standardize_descriptive=self.standardize_descriptive,
                                   standardize_clustering=self.standardize_clustering, rng=rng,
                                   objective=self.objective)

        feature_importance = np.zeros(root_data.d-1)
        if num_features == root_data.d-1:
            features = np.arange(root_data.d).astype(np.intp)

        root_data.calc_impurity(self.eps)
        root_node = Node(0)
        splitting_queue = [(root_node, root_data, root_data.min_labelled())]
        node_list = [root_node]
        while splitting_queue:
            node, data, num_labelled = splitting_queue.pop()
            successful_split = False
            if data.total_impurity(clustering_weights) > 2 * self.eps and \
                    node.depth < self.max_depth and \
                    num_labelled >= self.min_examples_to_split:

                # Try to split the node
                if num_features < data.d-1:
                    features = self.rng.choice(data.d-1, size=num_features+1, replace=False).astype(np.intp)
                    features[-1] = data.d-1
                    features.sort()
                splitter.learn_split(data, features)
                if data.descriptive_data.is_sparse:
                    split_weights = memview_to_SMatrix(splitter.weights_bias, data.d, features)
                else:
                    split_weights = memview_to_DMatrix(splitter.weights_bias, data.d, features)

                data_right, data_left = data.split(split_weights, splitter.threshold)
                labelled_left = data_left.min_labelled()
                labelled_right = data_right.min_labelled()

                if labelled_left > 0 and labelled_right > 0:
                    data_left.calc_impurity(self.eps)
                    data_right.calc_impurity(self.eps)

                    if relative_impurity(data_left, data, clustering_weights) < self.max_relative_impurity or \
                            relative_impurity(data_right, data, clustering_weights) < self.max_relative_impurity:
                        # We have a useful split!
                        feature_importance[features[:-1]] += splitter.feature_importance

                        if data.descriptive_data.is_sparse:
                            node.split_weights = memview_to_SMatrix(splitter.weights_bias[:-1], data.d-1, features[:-1])
                        else:
                            node.split_weights = memview_to_DMatrix(splitter.weights_bias[:-1], data.d-1, features[:-1])
                        node.threshold = splitter.threshold - splitter.weights_bias[-1]
                        left_node = Node(depth=node.depth + 1)
                        node_list.append(left_node)
                        node.left = len(node_list) - 1
                        right_node = Node(depth=node.depth + 1)
                        node_list.append(right_node)
                        node.right = len(node_list) - 1
                        if data.missing_descriptive:
                            feature_means = np.zeros(data.d, dtype=DTYPE)
                            feature_means[features] = splitter.d_means
                            node.feature_means = feature_means
                        successful_split = True
                        splitting_queue.append((right_node, data_right, labelled_right))
                        splitting_queue.append((left_node, data_left, labelled_left))

            if not successful_split:
                # Turn the node into a leaf
                if False and self.sparse_target:
                    temp = np.empty(data.t, dtype=DTYPE)
                    data.target_data.column_means(temp)
                    node.prototype = sp.csr_matrix(temp.reshape(1, -1))
                elif data.missing_target:
                    node.prototype = np.empty(data.t, dtype=DTYPE)
                    data.target_data.column_means_nan(node.prototype)
                else:
                    node.prototype = np.empty(data.t, dtype=DTYPE)
                    data.target_data.column_means(node.prototype)

        iterations = splitter.total_iterations

        return np.array(node_list), feature_importance, iterations

    # @staticmethod
    # def _traverse_tree(node_list, data_matrix, example_row):
    #     node = node_list[0]
    #     while not node.is_leaf():
    #         s = node.test(data_matrix, example_row)
    #         if s <= node.threshold:
    #             node = node_list[node.left]
    #         else:
    #             node = node_list[node.right]
    #     return node.prototype

    def split_weight_stats(self):
        small = 0
        tiny = 0
        nodes = 0
        features = 0
        for tree in self.trees:
            for node in tree:
                if node.split_weights is not None:
                    a = np.abs(node.split_weights.to_ndarray()[0])
                    features = a.shape[0]
                    nodes += 1
                    small += np.sum(a <= 1e-4)
                    tiny += np.sum(a <= 1e-8)

        return small, tiny, nodes, features
