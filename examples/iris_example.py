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

# Standardize the features (not obligatory, but recommended).
means = X_train.mean(axis=0)
stds = X_train.std(axis=0)
X_train = (X_train - means) / stds
X_test = (X_test - means) / stds

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
