import numpy as np
from randomforest.RandomForest import RandomForest
from randomforest.DecisionTree import DecisionTree

def test_simple_data():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = RandomForest(n_trees=5, max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert (preds == y).all()

def test_wine_quality_subset():
    X = np.array([[7.4,0.7,0.0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]])
    y = np.array([5])
    model = RandomForest(n_trees=3)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred[0] == 5
