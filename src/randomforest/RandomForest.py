#from DecisionTree import DecisionTree
from .DecisionTree import DecisionTree
import numpy as np
from collections import Counter


class RandomForest:
    """
    A Random Forest Classifier implementation.

    Parameters:
        n_trees : int, default=50
            Number of trees in the forest.
        max_depth : int, default=50
            Maximum depth of each tree.
        min_samples_split : int, default=2
            Minimum samples required to split a node.
        n_features : int, optional
            Number of features to consider for best split.
    """
    def __init__(self, n_trees=50, max_depth=50, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """
        Fit the random forest model.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Training data.
            y : array-like of shape (n_samples)
                Target values.
        """
        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _most_common_label(self, labels):
        counter = Counter(labels)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        Predict class for X.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Input samples.

        Returns:
            predictions : array-like of shape (n_samples)
                Predicted values.
        """
        all_predictions = []

        for i in range(len(X)):
            sample = X[i]
            votes = []

            for tree in self.trees:
                prediction = tree.predict([sample])[0]
                votes.append(prediction)

            final_prediction = self._most_common_label(votes)
            all_predictions.append(final_prediction)

        return np.array(all_predictions)
