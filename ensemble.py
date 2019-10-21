import torch
from tree import PCT
import numpy as np


class Ensemble:

    def __init__(self, num_models=10, bootstrapping=True, subspace_size=1, max_depth=np.inf, minimum_examples_to_split=2,
                 device='cpu', epochs=10, bs=None, lr=0.01):
        self.num_models = num_models
        self.bootstrapping = bootstrapping
        self.minimum_examples_to_split = minimum_examples_to_split
        self.num_nodes = 0
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.max_depth = max_depth
        self.subspace_size = subspace_size
        self.trees = None
        self.num_targets = 0

    def fit(self, data, descriptive_attributes, target_attributes, clustering_attributes=None):
        self.trees = []
        self.num_targets = target_attributes.shape[0]
        for _ in range(self.num_models):
            tree = PCT(max_depth=self.max_depth, subspace_size=self.subspace_size,
                       minimum_examples_to_split=self.minimum_examples_to_split,
                       device=self.device, epochs=self.epochs, bs=self.bs, lr=self.lr)
            if self.bootstrapping:
                rows = torch.randint(data.shape[0], size=(data.shape[0],))
            else:
                rows = None
            tree.fit(data, descriptive_attributes, target_attributes, clustering_attributes, rows)
            self.trees.append(tree)

    def predict(self, data):
        predictions = torch.zeros(data.shape[0], self.num_targets)
        for tree in self.trees:
            predictions += tree.predict(data)
        return predictions / self.num_models
