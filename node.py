
class Node:

    def __init__(self, depth=0):
        self.left = None
        self.right = None
        self.prototype = None
        self.split_weights = None
        self.split_bias = None
        self.order = None
        self.depth = depth

    def predict(self, x):
        if self.is_leaf():
            return self.prototype
        elif x.dot(self.split_weights) <= -self.split_bias:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def is_leaf(self):
        return self.left is None and self.right is None
