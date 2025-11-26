import numpy as np
from collections import Counter


class Node:
    """
    Represents a single node in the Decision Tree.

    Parameters:
        feature : int, optional
            Index of the feature used for splitting this node.
        threshold : float, optional
            Threshold value for the split.
        left : Node, optional
            Left child node.
        right : Node, optional
            Right child node.
        value : any, optional
            Class label value if this is a leaf node.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Check if the node is a leaf (has a value)."""
        return self.value is not None


class DecisionTree:
    """
    A decision Tree Classifier.

    Parameters:
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node.
        max_depth : int, default=100
            The maximum depth of the tree.
        n_features : int, optional
            The number of features to consider when looking for the best split.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters:
            X : array-like of shape (n_samples, n_features)
                The training input samples.
            y : array-like of shape (n_samples)
                The target values (class labels)
        """
        if self.n_features is None:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        features_indices = np.random.choice(n_features, self.n_features, replace=False)

        best_feature, best_threshold = self._best_split(X, y, features_indices)

        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)

        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in feature_indices:
            X_column = X[:, feature]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, X_column, split_threshold):
        parent_entropy = self._entropy(y)

        left_indices, right_indices = self._split(X_column, split_threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, threshold):
        left_indices = []
        right_indices = []
        for i in range(len(X_column)):
            if X_column[i] <= threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)
        return left_indices, right_indices

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        entropy = 0
        for p in ps:
            if p > 0:
                entropy -= p * np.log(p)
        return entropy

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        Predict class for X.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            y : array-like of shape (n_samples)
                The predicted classes.
        """
        predictions = []
        for i in range(len(X)):
            prediction = self._traverse_tree(X[i], self.root)
            predictions.append(prediction)
        return np.array(predictions)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
