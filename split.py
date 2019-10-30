import torch
import math
import numpy as np


# class SparseLinear(torch.nn.Module):
#     def __init__(self, in_size):
#         super(SparseLinear, self).__init__()
#         self.weights = torch.nn.sparse


def learn_split(rows, descriptive_data, clustering_data, device, epochs, bs, lr, subspace_size):
    selected_attributes = np.random.choice(a=[False, True],
                                           size=descriptive_data.size(1),
                                           p=[1-subspace_size, subspace_size])
    descriptive_subset = descriptive_data[:, selected_attributes]
    model = torch.nn.Linear(in_features=descriptive_subset.size(1), out_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if bs is None:
        bs = rows.size(0)
    num_batches = math.ceil(rows.size(0) / bs)

    for e in range(epochs):
        for b in range(num_batches):
            descr = descriptive_subset[b*bs:(b+1)*bs]
            clustr = clustering_data[b*bs:(b+1)*bs]
            right_selection = model(descr).reshape(-1).sigmoid()
            left_selection = torch.tensor(1., device=device) - right_selection
            right_weight_sum = torch.sum(right_selection)
            left_weight_sum = torch.sum(left_selection)
            var_left = weighted_variance(clustr, left_selection, left_weight_sum)
            var_right = weighted_variance(clustr, right_selection, right_weight_sum)
            impurity = (left_weight_sum * var_left + right_weight_sum * var_right)#+ torch.norm(model.weight, p=0.5)
            impurity.backward()
            optimizer.step()
            model.zero_grad()

    model.weight.require_grad = False
    model.bias.require_grad = False
    split_model = torch.nn.Linear(in_features=descriptive_data.size(1), out_features=1).to(device).eval()
    split_model.weight.requires_grad = False
    split_model.bias.requires_grad = False
    split_model.weight.zero_()
    split_model.weight[0][selected_attributes] = model.weight
    split_model.bias = model.bias
    return split_model


def weighted_variance(values, weights, weight_sum):
    mean = torch.matmul(weights, values) / weight_sum
    return -torch.sum(mean*mean)
