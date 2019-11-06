from tree import PCT
import numpy as np


class Ensemble:

    def __init__(self,
                 num_models=10,
                 bootstrapping=True,
                 max_depth=np.inf,
                 subspace_size=1,
                 minimum_examples_to_split=2,
                 epochs=10,
                 lr=0.01,
                 adam_params=(0.9, 0.999, 1e-8)):
        self.num_models = num_models
        self.bootstrapping = bootstrapping
        self.max_depth = max_depth
        self.subspace_size = subspace_size
        self.minimum_examples_to_split = minimum_examples_to_split
        self.epochs = epochs
        self.lr = lr
        self.adam_params = adam_params
        self.trees = None
        self.num_targets = 0
        self.num_nodes = 0

    def fit(self, descriptive_data, target_data, clustering_data=None,
            sparse_descriptive=False, sparse_target=False, sparse_clustering=False):
        self.trees = []
        self.num_targets = target_data.shape[1]
        for _ in range(self.num_models):
            tree = PCT(max_depth=self.max_depth, subspace_size=self.subspace_size,
                       minimum_examples_to_split=self.minimum_examples_to_split,
                       epochs=self.epochs, lr=self.lr, adam_params=self.adam_params)
            if self.bootstrapping:
                rows = np.random.randint(target_data.shape[0], size=(target_data.shape[0],))
            else:
                rows = None
            tree.fit(descriptive_data, target_data, clustering_data, rows,
                     sparse_descriptive, sparse_target, sparse_clustering)
            self.num_nodes += tree.num_nodes
            self.trees.append(tree)

    def predict(self, data):
        predictions = np.zeros((data.shape[0], self.num_targets))
        for tree in self.trees:
            predictions += tree.predict(data)
        return predictions / self.num_models
