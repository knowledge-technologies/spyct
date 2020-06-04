import numpy as np
from sklearn.svm._liblinear import train_wrap, set_verbosity_wrap
from spyct.clustering import kmeans


class SVMSplitter:

    def __init__(self, n, d, c, clustering_weights, opt_iter, cluster_iter, eps, C, tol,
                 balance_classes, standardize_descriptive, standardize_clustering, rng, objective):
        self.n = n
        self.d = d
        self.c = c
        self.clustering_weights = clustering_weights
        self.opt_iter = opt_iter
        self.cluster_iter = cluster_iter
        self.eps = eps
        self.C = C
        self.tol = tol
        self.balance_classes = balance_classes
        self.class_weight = 'balanced' if balance_classes else None
        self.standardize_descriptive = standardize_descriptive
        self.standardize_clustering = standardize_clustering
        self.rng = rng

        self.feature_importance = np.zeros(d-1)
        self.total_iterations = 0
        self.weights_bias = None
        self.threshold = 0

        self.d_means = np.empty(d, dtype='f')
        self.d_stds = np.empty(d, dtype='f')
        self.c_means = np.empty(c, dtype='f')
        self.c_stds = np.empty(c, dtype='f')

        set_verbosity_wrap(0)

        self.distance = None
        self.objective = objective

    def learn_split(self, data, features):

        descriptive_data = data.descriptive_data.take_columns(features)
        clustering_data = data.clustering_data.copy()

        # if the data is not sparse, standardize it and handle missing values
        if self.standardize_descriptive:
            if data.missing_descriptive:
                descriptive_data.column_stds_nan(1, self.d_means, self.d_stds)
                self.d_means[self.d-1] = 0
                descriptive_data.standardize_columns(self.d_means, self.d_stds)
                descriptive_data.impute_missing(0)
            else:
                descriptive_data.column_stds(1, self.d_means, self.d_stds)
                self.d_means[self.d-1] = 0
                descriptive_data.standardize_columns(self.d_means, self.d_stds)

        if self.standardize_clustering:
            if data.missing_clustering:
                clustering_data.column_stds_nan(1, self.c_means, self.c_stds)
            else:
                clustering_data.column_stds(1, self.c_means, self.c_stds)

            if self.clustering_weights is not None:
                self.c_stds /= self.clustering_weights

            clustering_data.standardize_columns(self.c_means, self.c_stds)
            if data.missing_clustering:
                clustering_data.impute_missing(0)

        left_or_right = self.cluster(clustering_data)
        s = np.sum(left_or_right)
        if s == 0 or s == left_or_right.shape[0]:
            # We have but one cluster, no splitting to do.
            # Try with different starting centroids
            left_or_right = self.cluster(clustering_data)
        if s == 0 or s == left_or_right.shape[0]:
            # If we fail again, just stop.
            self.weights_bias = np.zeros(self.d, dtype='f')
            return

        if self.balance_classes:
            class_weights = np.array([left_or_right.shape[0] - s, s], dtype='d', order='C')
        else:
            class_weights = np.ones((2,), dtype='d', order='C')

        if descriptive_data.is_sparse:
            features = descriptive_data.to_csr()
        else:
            features = descriptive_data.to_ndarray().astype('d', order='C')

        solver_type = 5  # 5=squared_hinge (svm);  6=logistic
        coef, n_iter = train_wrap(features, left_or_right, descriptive_data.is_sparse, solver_type,
                                  self.tol, -1, self.C, class_weights, self.opt_iter, self.rng.randint(9999999),
                                  self.eps, np.ones(left_or_right.shape[0]))

        self.weights_bias = coef.squeeze().astype('f')
        self.total_iterations += n_iter[0]

        # Feature importance update
        support = descriptive_data.n_rows / self.n
        norm = np.linalg.norm(self.weights_bias[:-1], ord=1) + self.eps
        self.weights_bias[np.abs(self.weights_bias) < 2*self.eps] = 0
        self.feature_importance = support * np.abs(self.weights_bias[:-1]) / norm

        if self.standardize_descriptive:
            self.weights_bias /= self.d_stds
            if not descriptive_data.is_sparse:
                self.threshold = self.weights_bias.dot(self.d_means)

    def cluster(self, clustering_data):
        distance_code = 1 if self.objective == 'mse' else 2
        r0 = self.rng.randint(clustering_data.n_rows)
        r1 = 0
        while clustering_data.equal_rows(r0, r1):
            r1 += 1

        left_or_right = kmeans(clustering_data, clustering_data.row_vector(r0), clustering_data.row_vector(r1),
                               self.cluster_iter, self.tol, self.eps, distance_code)
        return np.asarray(left_or_right, dtype='d')
