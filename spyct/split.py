import numpy as np
import scipy.sparse as sp


def derivative(weights_bias, descriptive_values, clustering_values, eps):
    n, d = descriptive_values.shape
    t = descriptive_values.dot(weights_bias[:-1]) + weights_bias[-1]
    exps = np.exp(-t)
    exps_1 = exps + 1
    right_selection = 1 / exps_1
    left_selection = 1 - right_selection
    right_total = np.sum(right_selection) + eps
    left_total = n - right_total + eps

    if sp.isspmatrix(clustering_values):
        right_weighted_sums = sp.csr_matrix.dot(right_selection, clustering_values)
        left_weighted_sums = sp.csr_matrix.dot(left_selection, clustering_values)
    else:
        right_weighted_sums = right_selection.dot(clustering_values)
        left_weighted_sums = left_selection.dot(clustering_values)

    der_y_by_selection = 2 * clustering_values.dot(right_weighted_sums) / right_total - \
                         np.sum(right_weighted_sums * right_weighted_sums) / (right_total * right_total) - \
                         2 * clustering_values.dot(left_weighted_sums) / left_total + \
                         np.sum(left_weighted_sums * left_weighted_sums) / (left_total * left_total)

    der_selection_by_bias = exps / (exps_1 * exps_1)

    if sp.isspmatrix(descriptive_values):
        der_selection_by_weights = descriptive_values.multiply(der_selection_by_bias.reshape(-1, 1))
    else:
        der_selection_by_weights = der_selection_by_bias.reshape(-1, 1) * descriptive_values

    derivatives = np.zeros(weights_bias.shape)
    derivatives[:-1] = der_selection_by_weights.transpose().dot(der_y_by_selection)
    derivatives[-1] = np.dot(der_selection_by_bias, der_y_by_selection)
    return derivatives


def learn_split(descriptive_data, clustering_data, epochs, lr, subspace_size,
                adam_params=(0.9, 0.999, 1e-8)):
    selected_attributes = np.random.choice(a=[False, True],
                                           size=descriptive_data.shape[1],
                                           p=[1 - subspace_size, subspace_size])
    beta1, beta2, eps = adam_params
    descriptive_subset = descriptive_data[:, selected_attributes]

    std = 1 / np.sqrt(descriptive_subset.shape[1])
    weights_bias = -std + 2 * std * np.random.rand(descriptive_subset.shape[1] + 1)
    moments1 = np.zeros(weights_bias.shape)
    moments2 = np.zeros(weights_bias.shape)
    beta1t = 1
    beta2t = 1
    for e in range(epochs):
        grad = derivative(weights_bias, descriptive_subset, clustering_data, eps)

        # Adam
        beta1t *= beta1
        beta2t *= beta2
        moments1 = beta1 * moments1 + (1 - beta1) * grad
        moments2 = beta2 * moments2 + (1 - beta2) * grad * grad
        m1 = moments1 / (1 - beta1t)
        m2 = moments2 / (1 - beta2t)
        weights_bias += lr * m1 / (np.sqrt(m2) + eps)

    weights = np.zeros(descriptive_data.shape[1])
    weights[selected_attributes] = weights_bias[:-1]
    return weights, weights_bias[-1]
