import numpy as np
from scipy.special import expit


# @njit
def derivative(weights_bias, descriptive_values, clustering_values, eps):

    n, d = descriptive_values.shape
    scores = np.matmul(descriptive_values, weights_bias[:-1]) + weights_bias[-1]
    sigmoided = expit(scores)
    S = np.sum(sigmoided) + eps
    S_neg = n - S + eps
    weighted_sums = np.matmul(sigmoided, clustering_values)
    weighted_sums_neg = np.matmul(1-sigmoided, clustering_values)

    Y_s = 2 * np.matmul(clustering_values, weighted_sums) / S \
          - np.sum(weighted_sums * weighted_sums) / (S*S) \
          - 2 * np.matmul(clustering_values, weighted_sums_neg) / S_neg \
          + np.sum(weighted_sums_neg * weighted_sums_neg) / (S_neg * S_neg)

    exps = np.exp(-scores)
    exps_1 = exps + 1
    s_b = exps / (exps_1 * exps_1)
    s_w = s_b.reshape(-1, 1) * descriptive_values

    derivatives = np.zeros(weights_bias.shape)
    derivatives[:-1] = np.matmul(s_w.transpose(), Y_s)
    derivatives[-1] = np.dot(s_b, Y_s)
    return derivatives


def learn_split(descriptive_data, clustering_data, epochs, lr, subspace_size, adam_params):
    selected_attributes = np.random.choice(a=[False, True],
                                           size=descriptive_data.shape[1],
                                           p=[1 - subspace_size, subspace_size])
    beta1, beta2, eps = adam_params
    descriptive_subset = descriptive_data[:, selected_attributes]
    weights_bias = np.random.randn(descriptive_subset.shape[1]+1)
    moments1 = np.zeros(weights_bias.shape)
    moments2 = np.zeros(weights_bias.shape)
    beta1t = 1
    beta2t = 1
    for e in range(epochs):
        grad = derivative(weights_bias, descriptive_subset, clustering_data, eps)

        # SGD
        # weights_bias += lr * grad

        # Adam
        beta1t *= beta1
        beta2t *= beta2
        moments1 = beta1 * moments1 + (1-beta1) * grad
        moments2 = beta2 * moments2 + (1-beta2) * grad*grad
        m1 = moments1 / (1 - beta1t)
        m2 = moments2 / (1 - beta2t)
        weights_bias += lr * m1 / (np.sqrt(m2) + eps)

    weights = np.zeros(descriptive_data.shape[1])
    weights[selected_attributes] = weights_bias[:-1]
    return weights, weights_bias[-1]
