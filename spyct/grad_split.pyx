from libc.math cimport exp, sqrt, fabs, isnan
from spyct._math cimport *
from spyct._matrix cimport *
from spyct.data cimport Data
import numpy as np


cdef:
    str fmt = "f"
    int DOT_OBJECTIVE = 0
    int MSE_OBJECTIVE = 1


cdef class GradSplitter:

    cdef:
        # parameters
        index n, d, c, max_iter
        DTYPE eps, tol, beta1, beta2, regularization, lr
        DTYPE[::1] clustering_weights
        bint standardize_descriptive, standardize_clustering
        object rng
        int objective

        # stats
        readonly index total_iterations
        readonly DTYPE[::1] feature_importance
        DTYPE score

        # single purpose arrays
        readonly DTYPE[::1] d_means # needed when making predictions with missing features
        DTYPE[::1] grad, moments1, moments2, d_stds, c_means, c_stds
        Matrix clustering_nonmissing

        # multi purpose arrays
        DTYPE[::1] vec_c1, vec_c2, vec_c3, vec_c4, vec_c5, vec_n1, vec_n2, vec_n3

        # split
        readonly DTYPE[::1] weights_bias
        readonly DTYPE threshold

    def __init__(self, int n, int d, int c,
                 DTYPE[::1] clustering_weights, int max_iter, DTYPE learning_rate,
                 DTYPE regularization, DTYPE adam_beta1, DTYPE adam_beta2,
                 DTYPE eps, DTYPE tol, bint standardize_descriptive,
                 bint standardize_clustering, object rng, str objective):
        # parameters
        self.n = n
        self.d = d
        self.c = c
        self.clustering_weights = clustering_weights
        self.max_iter = max_iter
        self.lr = learning_rate
        self.regularization = regularization
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2
        self.eps = eps
        self.tol = tol
        self.standardize_descriptive = standardize_descriptive
        self.standardize_clustering = standardize_clustering
        self.rng = rng

        self.objective = MSE_OBJECTIVE
        if objective == 'dot':
            self.objective = DOT_OBJECTIVE
            raise ValueError('Only mse objective supported for grad splitter')

        # statistics
        self.total_iterations = 0
        self.threshold = 0
        self.score = 0
        self.feature_importance = create_real_vector(d-1)
        reset_vector(self.feature_importance, 0)

        # arrays we can reuse for every split
        self.weights_bias = create_real_vector(d)
        self.grad = create_real_vector(d)
        self.moments1 = create_real_vector(d)
        self.moments2 = create_real_vector(d)
        self.d_means = create_real_vector(d)
        self.d_stds = create_real_vector(d)

        self.c_means = create_real_vector(c)
        self.c_stds = create_real_vector(c)
        self.vec_c1 = create_real_vector(c)
        self.vec_c2 = create_real_vector(c)
        self.vec_c3 = create_real_vector(c)
        self.vec_c4 = create_real_vector(c)
        self.vec_c5 = create_real_vector(c)

        # Arrays we can reuse for each gradient calculation, but not for different splits
        self.clustering_nonmissing = None
        self.vec_n1 = None
        self.vec_n2 = None
        self.vec_n3 = None


    cpdef void learn_split(self, Data data, index[::1] features):

        cdef:
            index e, row, col, n, i
            DTYPE previous_score, beta1t, beta2t, norm, support, delta, old_w, g
            Matrix descriptive_data = data.descriptive_data.take_columns(features)
            Matrix clustering_data = data.clustering_data.copy()

        n = descriptive_data.n_rows
        self.vec_n1 = create_real_vector(n)
        self.vec_n2 = create_real_vector(n)
        self.vec_n3 = create_real_vector(n)

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
                component_div(self.c_stds, self.clustering_weights, self.c_stds)

            clustering_data.standardize_columns(self.c_means, self.c_stds)

            if data.missing_clustering:
                self.clustering_nonmissing = clustering_data.nonmissing_matrix()
                clustering_data.impute_missing(0)

        # splits the points in half
        self.weights_bias = self.rng.normal(loc=0, scale=self.eps, size=self.d).astype(fmt)
        self.weights_bias[self.d-1] = 0
        projections = self.vec_n1
        descriptive_data.self_dot_vector(self.weights_bias, projections)
        self.weights_bias[self.d-1] = -np.median(projections)
        l2_normalize(self.weights_bias)

        # optimization
        reset_vector(self.moments1, 0)
        reset_vector(self.moments2, 0)
        self.threshold = 0
        self.score = 0
        previous_score = -1e+10
        beta1t = 1
        beta2t = 1
        for e in range(self.max_iter):
            self.total_iterations += 1

            if self.objective == MSE_OBJECTIVE:
                self._variance_derivative(descriptive_data, clustering_data, data.missing_clustering)

            # regularization
            norm = l05_norm(self.weights_bias) / self.d
            # norm = l1_norm(self.weights_bias) / self.d
            self.score -= self.regularization * norm
            for i in range(self.d-1):
                old_w = self.weights_bias[i]
                g = sqrt(norm / fabs(old_w))
                if old_w > 0:
                    self.grad[i] -= self.regularization * g
                    # self.grad[i] -= self.regularization / self.d
                elif old_w < 0:
                    self.grad[i] += self.regularization * g
                    # self.grad[i] += self.regularization / self.d
            # self.score /= n * self.c
            # l2_normalize(self.grad)

            # Adam
            beta1t *= self.beta1
            beta2t *= self.beta2
            norm = 0
            for col in range(self.d):
                g = self.grad[col]
                self.moments1[col] = self.beta1 * self.moments1[col] + (1 - self.beta1) * g
                self.moments2[col] = self.beta2 * self.moments2[col] + (1 - self.beta2) * g * g
                delta = self.lr * (self.moments1[col] / (1-beta1t)) / (sqrt(self.moments2[col] / (1-beta2t)) + self.eps)
                old_w = self.weights_bias[col]
                # sign is flipping
                if old_w > 0 and old_w + delta < 0:
                    delta = - old_w - self.eps
                elif old_w < 0 and old_w + delta > 0:
                    delta = - old_w + self.eps

                norm += abs(delta)
                self.weights_bias[col] += delta

            # early stopping
            if norm / (self.d * self.lr)  < self.tol:
                break

            # if self.score - previous_score < self.tol:
            #     break
            # else:
            #     previous_score = self.score

        # Weights below 2*self.eps are put to 0
        # Normalize the final weights and update feature importances
        support = (<DTYPE>n) / self.n
        norm = l1_norm(self.weights_bias[:self.d-1]) + self.eps
        for col in range(self.d):
            g = fabs(self.weights_bias[col])
            if g < 2*self.eps:
                self.weights_bias[col] = 0
            else:
                self.weights_bias[col] /= norm
                if col < self.d-1:
                    self.feature_importance[col] = support * g / norm

        if self.standardize_descriptive:
            component_div(self.weights_bias, self.d_stds, self.weights_bias)
            if not descriptive_data.is_sparse:
                self.threshold = vector_dot_vector(self.weights_bias, self.d_means)


    cpdef void _variance_derivative(self, Matrix descriptive_values, Matrix clustering_values, bint missing_clustering):
        cdef:
            DTYPE[::1] right_selection, left_selection, right_nonmissing, left_nonmissing
            DTYPE[::1] selection_derivative, right_p, left_p, right_p_sq, left_p_sq, diff_p, diff_p_sq
            DTYPE[::1] der_y_by_selection, temp
            DTYPE num_left, num_right

        right_selection = self.vec_n1
        left_selection = self.vec_n2
        selection_derivative = self.vec_n3
        descriptive_values.self_dot_vector(self.weights_bias, right_selection)
        fuzzy_split_sigmoid(right_selection, left_selection, right_selection, selection_derivative)

        if missing_clustering:
            right_nonmissing = self.vec_c1
            left_nonmissing = self.vec_c2
            self.clustering_nonmissing.vector_dot_self(right_selection, right_nonmissing)
            vector_scalar_sum(right_nonmissing, self.eps)
            self.clustering_nonmissing.vector_dot_self(left_selection, left_nonmissing)
            vector_scalar_sum(left_nonmissing, self.eps)
        else:
            num_left = vector_sum(left_selection) + self.eps
            num_right = vector_sum(right_selection) + self.eps

        right_p = self.vec_c3
        left_p = self.vec_c4
        clustering_values.vector_dot_self(right_selection, right_p)
        clustering_values.vector_dot_self(left_selection, left_p)
        if missing_clustering:
            component_div(right_p, right_nonmissing, right_p)
            component_div(left_p, left_nonmissing, left_p)
        else:
            vector_scalar_prod(right_p, 1 / num_right)
            vector_scalar_prod(left_p, 1 / num_left)

        diff_p = self.vec_c5
        right_p_sq = right_p
        left_p_sq = left_p
        component_diff(right_p, left_p, self.vec_c5)
        component_prod(right_p, right_p, right_p_sq)
        component_prod(left_p, left_p, left_p_sq)

        # if missing_clustering:
        #     self.score = vector_dot_vector(right_nonmissing, right_p_sq) + vector_dot_vector(left_nonmissing, left_p_sq)
        # else:
        #     self.score = num_left * vector_sum(left_p_sq) + num_right * vector_sum(right_p_sq)

        diff_p_sq = self.vec_c3
        der_y_by_selection = self.vec_n1
        temp = self.vec_n2
        clustering_values.self_dot_vector(diff_p, der_y_by_selection)
        vector_scalar_prod(der_y_by_selection, 2)
        component_diff(right_p_sq, left_p_sq, diff_p_sq)

        if missing_clustering:
            self.clustering_nonmissing.self_dot_vector(diff_p_sq, temp)
            component_diff(der_y_by_selection, temp, der_y_by_selection)
        else:
            vector_scalar_sum(der_y_by_selection, -vector_sum(diff_p_sq))

        component_prod(der_y_by_selection, selection_derivative, der_y_by_selection)
        descriptive_values.vector_dot_self(der_y_by_selection, self.grad)
