A python implementation of multivariate predictive clustering trees.

# Features
- Support for various predictive modelling tasks 
(binary, multi-class, multi-label, hierarchical classification, single- and multi-target regression).
- Supervised and semi-supervised learning.
- Handles missing values in the data seamlessly.

# Installation
You can install the package directly from the git repository:

`pip install git+https://gitlab.com/TStepi/spyct.git`

# Dependencies
- `numpy`
- `scipy`
- `scikit-learn`
- `joblib`
- a C compiler (e.g., `gcc`)

# Usage example
```
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import spyct


# Load the iris dataset from scikit learn.
dataset = load_iris()
X = dataset.data
y = dataset.target.reshape(-1, 1)


# Split the data into train and test subsets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# For multi-class datasets, target needs to be one-hot encoded.
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train).toarray()

# Fit the model and make predictions
model = spyct.Model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# decode the encoded predictions
y_pred = np.argmax(y_pred, axis=1)

# Show the confusion matrix
print(confusion_matrix(y_test, y_pred))
```

# Citation
```
@misc{stepinik2020oblique,
    title={Oblique Predictive Clustering Trees},
    author={Tomaž Stepišnik and Dragi Kocev},
    year={2020},
    eprint={2007.13617},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

# Model parameters
Brief description of parameters of the `Model` class. For details see the paper.

- `splitter: string, (default="grad")`

    Determines which split optimizer to use. Supported values are `"grad"` and `"svm"`.


-  `num_trees: int, (default=100).`

    The number of trees in the model.


- `max_features: int, float, "sqrt", "log" (default=1.0)`

    The number of features to consider when optimizing the splits:
    - If `int`, then consider `max_features` features at each split.
    - If `float`, then `max_features` is a fraction and
        `int(max_features * n_features)` features are considered at each
        split.
    - If `"sqrt"`, then `max_features=sqrt(n_features)`.
    - If `"log"`, then `max_features=log2(n_features)`.

    At least one feature is always considered.


- `bootstrapping: boolean, (default=None)`

    Whether to use bootstrapped samples of the learning set to train each
    tree. If not set, bootstrapping is used when learning more than one tree.


- `max_depth: int, (default=inf)`

    The maximum depth the trees can grow to. Unlimited by default.


- `min_examples_to_split: int, (default=2)`

    Minimum number of examples required to split an internal node. When the number of examples falls below this
    threshold, a leaf node is made.


- `min_impurity_decrease: float, (default=0)`

    Minimum relative impurity decrease of at least one subset produced by a split. If not achieved, the
    splitting stops and a leaf node is made.


- `n_jobs: int, (default=1)`

    The number of parallel jobs to use when building a forest. Uses process based parallelism with joblib.


- `standardize_descriptive: boolean, (default=True)`

    Determines if the descriptive data is standardized to mean=0 and std=1 when learning weights for each split.
    If the data is sparse, mean is assumed to be 0, to preserve sparsity.


- `standardize_clustering: boolean, (default=True)`

    Determines if the clustering data is standardized to mean=0 and std=1 when learning weights for each split.
    If the data is sparse, mean is assumed to be 0, to preserve sparsity.


- `max_iter: int, (default=100)`

    Maximum number of iterations a split is optimized for, if early stopping does not terminate the optimization
    beforehand.


- `lr: float, (default=0.01)`

    Learning rate used to optimize the splits in the `"grad"` splitter.


- `C: float, (default=0)`

    Split weight regularization parameter. The strength of the regularization is inversely proportional to `C`.


- `balance_classes: boolean, (default=True)`

    Used by the `"svm"` splitter. If True, automatically adjust weights of classes when learning the split to be
    inversely proportional to their frequencies in the data.


- `tol: float, (default=0)`

    Tolerance for stopping criteria.


- `eps: float, (default=1e-8)`

    A tiny value added to denominators for numeric stability where division by zero could occur.


- `adam_beta1: float, (default=0.9)`

    `Beta1` parameter for the Adam optimizer. Used by the `"grad"` splitter.


- `adam_beta2: float, (default=0.999)`

    `Beta2` parameter for the Adam optimizer. Used by the `"grad"` splitter.


- `random_state: RandomState instance, int, (default=None)`

    If provided, the `RandomState` instance will be used for any randomness. If provided an `int`, a `RandomState` instance
    with the provided `int` as the seed will be used.

