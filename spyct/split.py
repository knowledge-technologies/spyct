import numpy as np
import scipy.sparse as sp


def derivative(weights_bias, descriptive_values, clustering_values, eps):
    n, d = descriptive_values.shape
    t = descriptive_values.dot(weights_bias)
    exps = np.exp(-t)
    exps_1 = exps + 1
    right_selection = 1 / exps_1
    left_selection = 1 - right_selection
    right_total = np.sum(right_selection) + eps
    left_total = n - right_total + eps

    if sp.isspmatrix(clustering_values):
        right_weighted_sums = sp.csr_matrix.dot(right_selection, clustering_values) / right_total
        left_weighted_sums = sp.csr_matrix.dot(left_selection, clustering_values) / left_total
    else:
        right_weighted_sums = right_selection.dot(clustering_values) / right_total
        left_weighted_sums = left_selection.dot(clustering_values) / left_total

    right_var = np.sum(right_weighted_sums * right_weighted_sums)
    left_var = np.sum(left_weighted_sums * left_weighted_sums)

    der_y_by_selection = 2 * clustering_values.dot(right_weighted_sums) - right_var - \
                         2 * clustering_values.dot(left_weighted_sums) + left_var

    der_selection_by_bias = exps / (exps_1 * exps_1)
    if sp.isspmatrix(descriptive_values):
        der_y_by_weights = sp.csr_matrix.dot(der_y_by_selection * der_selection_by_bias, descriptive_values)
    else:
        der_y_by_weights = (der_y_by_selection * der_selection_by_bias).dot(descriptive_values)

    return der_y_by_weights, right_total*right_var + left_total*left_var


def learn_split(descriptive_data, clustering_data, epochs, lr, adam_params, early_stopping_params):

    beta1, beta2, eps = adam_params
    patience, delta = early_stopping_params

    # make arrays contiguous for faster calculations
    # TODO sparse?
    # descriptive_data = np.asarray(descriptive_data, order='C')
    # clustering_data = np.asarray(clustering_data, order='C')

    # initialize weights
    std = 1 / np.sqrt(descriptive_data.shape[1])
    weights_bias = -std + 2 * std * np.random.rand(descriptive_data.shape[1])

    # optimization
    moments1 = np.zeros(weights_bias.shape)
    moments2 = np.zeros(weights_bias.shape)
    beta1t = 1
    beta2t = 1
    previous_score = 0
    waiting = 0
    for _ in range(epochs):
        grad, score = derivative(weights_bias, descriptive_data, clustering_data, eps)

        # Adam
        beta1t *= beta1
        beta2t *= beta2
        moments1 = beta1 * moments1 + (1 - beta1) * grad
        moments2 = beta2 * moments2 + (1 - beta2) * grad * grad
        m1 = moments1 / (1 - beta1t)
        m2 = moments2 / (1 - beta2t)
        weights_bias += lr * m1 / (np.sqrt(m2) + eps)

        # early stopping
        if score < (1+delta) * previous_score:
            waiting += 1
            if waiting > patience:
                break
        else:
            previous_score = score
            waiting = 0

    return weights_bias
