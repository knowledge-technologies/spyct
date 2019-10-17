import torch
import math


def learn_split(data, descriptive_attributes, clustering_attributes, device, epochs, bs, lr):
    model = torch.nn.Linear(in_features=descriptive_attributes.size(0), out_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2 if lr is None else lr)
    descriptive_values = data[:, descriptive_attributes]
    clustering_values = data[:, clustering_attributes]
    clustering_sq_values = clustering_values * clustering_values
    if bs is None:
        bs = data.size(0)
    num_batches = math.ceil(data.size(0) / bs)

    for e in range(epochs):
        for b in range(num_batches):
            descr = descriptive_values[b*bs:(b+1)*bs]
            clustr = clustering_values[b*bs:(b+1)*bs]
            clustr_sq = clustering_sq_values[b*bs:(b+1)*bs]
            right_selection = model(descr).squeeze()
            right_selection.sigmoid_()
            left_selection = 1 - right_selection
            right_weight_sum = torch.sum(right_selection)
            left_weight_sum = torch.sum(left_selection)
            var_left = weighted_variance(clustr, clustr_sq, left_selection, left_weight_sum)
            var_right = weighted_variance(clustr, clustr_sq, right_selection, right_weight_sum)
            impurity = (left_weight_sum * var_left + right_weight_sum * var_right) / (left_weight_sum + right_weight_sum)
            impurity.backward()
            optimizer.step()
            model.zero_grad()

    return model.eval()


def weighted_variance(values, sq_values, weights, weight_sum):
    mean_sq = torch.matmul(weights, sq_values) / weight_sum
    mean = torch.matmul(weights, values) / weight_sum
    return torch.sum(mean_sq - mean*mean)
