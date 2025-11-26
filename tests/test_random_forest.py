import numpy as np
import pytest
from src.randomforest.RandomForest import RandomForest

def test_simple_classification():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    model = RandomForest(n_trees=10, max_depth=5)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.all(np.isin(preds, [0, 1]))

def test_wine_subset_execution():
    X = np.array([
        [7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4],
        [7.8, 0.88, 0.0, 2.6, 0.098, 25, 67, 0.9968, 3.2, 0.68, 9.8]
    ])
    y = np.array([5, 5])

    model = RandomForest(n_trees=2, max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 2

def test_parameters():
    model = RandomForest(n_trees=10, max_depth=3)
    assert model.n_trees == 10
    assert model.max_depth == 3
