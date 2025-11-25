import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from randomforest.RandomForest import RandomForest
from randomforest.DecisionTree import DecisionTree
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("winequality-red.csv")
X = data.drop("quality", axis=1).values
y = data["quality"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Our RandomForest
my_model = RandomForest(n_trees=20, max_depth=10)
my_model.fit(X_train, y_train)
my_predictions = my_model.predict(X_test)

my_accuracy = np.sum(my_predictions == y_test) / len(y_test)
print("Accuracy our implementation:", my_accuracy)

# scikit-learn RandomForest 
sk_model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=1)
sk_model.fit(X_train, y_train)
sk_predictions = sk_model.predict(X_test)

sk_accuracy = np.sum(sk_predictions == y_test) / len(y_test)
print("Accuracy scikit-learn:", sk_accuracy)

input("Press ENTER to exit")
