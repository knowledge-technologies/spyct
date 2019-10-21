import torch
import numpy as np
from node import Node
from split import learn_split


class PCT:

    def __init__(self, max_depth=np.inf, subspace_size=1, minimum_examples_to_split=2,
                 device='cpu', epochs=10, bs=None, lr=0.01):
        self.minimum_examples_to_split = minimum_examples_to_split
        self.root_node = None
        self.num_nodes = 0
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.max_depth = max_depth
        self.subspace_size = subspace_size

    def fit(self, data, descriptive_attributes, target_attributes, clustering_attributes=None, rows=None):

        data = torch.tensor(data, device=self.device, dtype=torch.float32)
        descriptive_attributes = torch.tensor(descriptive_attributes, device=self.device)
        target_attributes = torch.tensor(target_attributes, device=self.device)

        if clustering_attributes is None:
            clustering_attributes = target_attributes

        if rows is None:
            rows = torch.arange(data.shape[0])

        variances = torch.var(data, dim=0)
        total_variance = torch.sum(variances[clustering_attributes])

        self.root_node = Node(depth=0)
        splitting_queue = [(self.root_node, rows, total_variance)]
        order = 0
        while splitting_queue:
            node, rows, total_variance = splitting_queue.pop()
            node.order = order
            order += 1
            if total_variance > 0 and node.depth < self.max_depth and rows.size(0) >= self.minimum_examples_to_split:
                # Try to split the node
                split_model = learn_split(
                    rows, data, descriptive_attributes, clustering_attributes,
                    device=self.device, epochs=self.epochs, bs=self.bs, lr=self.lr, subspace_size=self.subspace_size)
                split = split_model(data[rows][:, descriptive_attributes]).squeeze()
                rows_right = rows[split > 0]
                var_right = torch.sum(torch.var(data[rows_right][:, clustering_attributes], dim=0))
                rows_left = rows[split <= 0]
                var_left = torch.sum(torch.var(data[rows_left][:, clustering_attributes], dim=0))
                if var_left < total_variance or var_right < total_variance:
                    # The split is useful. Apply it.
                    node.split_model = split_model
                    node.left = Node(depth=node.depth+1)
                    node.right = Node(depth=node.depth+1)
                    splitting_queue.append((node.left, rows_left, var_left, ))
                    splitting_queue.append((node.right, rows_right, var_right, ))
                else:
                    # Turn the node into a leaf
                    node.prototype = torch.mean(data[rows][:, target_attributes], dim=0)
            else:
                # Turn the node into a leaf
                node.prototype = torch.mean(data[rows][:, target_attributes], dim=0)
        self.num_nodes = order

    def predict(self, data):
        data = torch.tensor(data, device=self.device, dtype=torch.float32)
        raw_predictions = [self.root_node.predict(data[i]) for i in range(data.size(0))]
        return torch.stack(raw_predictions)
