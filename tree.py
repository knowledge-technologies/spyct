import torch
import numpy as np
from node import Node
from split import learn_split


class PCT:

    def __init__(self, max_depth=None, subspace_size=None, minimum_examples_to_split=2,
                 device='cpu', epochs=10, bs=None, lr=None):
        self.subspace_size = subspace_size
        self.minimum_examples_to_split = minimum_examples_to_split
        self.root_node = None
        self.num_nodes = 0
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        if max_depth is None:
            self.max_depth = np.inf
        else:
            self.max_depth = max_depth

        # Needed for prediction transformation
        # self.target_means = None
        # self.target_stds = None
        # self.descr_means = None
        # self.descr_stds = None

    def fit(self, data, descriptive_attributes, target_attributes, clustering_attributes=None):

        data = torch.tensor(data, device=self.device, dtype=torch.float32)
        descriptive_attributes = torch.tensor(descriptive_attributes, device=self.device)
        target_attributes = torch.tensor(target_attributes, device=self.device)

        if clustering_attributes is None:
            clustering_attributes = target_attributes

        variances, means = torch.var_mean(data, dim=0)
        total_variance = torch.sum(variances[clustering_attributes])
        # variances[variances == 0] = 1
        # data = (data - means) / torch.sqrt(variances)
        # self.target_means = means[target_attributes]
        # self.target_stds = torch.sqrt(variances[target_attributes])
        # self.descr_means = means[descriptive_attributes]
        # self.descr_stds = torch.sqrt(variances[descriptive_attributes])

        self.root_node = Node(depth=0)
        splitting_queue = [(self.root_node, data, total_variance)]
        order = 0
        while splitting_queue:
            node, data, total_variance = splitting_queue.pop()
            node.order = order
            order += 1
            if total_variance > 0 and node.depth < self.max_depth and data.size(0) >= self.minimum_examples_to_split:
                # Try to split the node
                split_model = learn_split(
                    data, descriptive_attributes, clustering_attributes,
                    device=self.device, epochs=self.epochs, bs=self.bs, lr=self.lr)
                split = split_model(data[:, descriptive_attributes]).squeeze()
                data_right = data[split > 0]
                var_right = torch.sum(torch.var(data_right[:, clustering_attributes], dim=0))
                data_left = data[split <= 0]
                var_left = torch.sum(torch.var(data_left[:, clustering_attributes], dim=0))
                if var_left < total_variance or var_right < total_variance:
                    # The split is useful. Apply it.
                    node.split_model = split_model
                    node.left = Node(depth=node.depth+1)
                    node.right = Node(depth=node.depth+1)
                    splitting_queue.append((node.left, data_left, var_left, ))
                    splitting_queue.append((node.right, data_right, var_right, ))
                else:
                    # Turn the node into a leaf
                    node.prototype = torch.mean(data[:, target_attributes], dim=0)
            else:
                # Turn the node into a leaf
                node.prototype = torch.mean(data[:, target_attributes], dim=0)
        self.num_nodes = order

    def predict(self, data):
        # data = (torch.tensor(data, device=self.device, dtype=torch.float32) - self.descr_means) / self.descr_stds
        data = torch.tensor(data, device=self.device, dtype=torch.float32)
        raw_predictions = [self.root_node.predict(data[i]) for i in range(data.size(0))]
        # return self.target_stds * (torch.stack(raw_predictions) + self.target_means)
        return torch.stack(raw_predictions)
