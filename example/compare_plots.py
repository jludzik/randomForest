from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.randomforest.RandomForest import RandomForest

def get_data():
    csv_path = os.path.join(os.path.dirname(__file__), "winequality-red.csv")
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}")
        data = pd.read_csv(csv_path)
        X = data.drop("quality", axis=1).values
        y = data["quality"].values
    else:
        print("CSV not found. Loading sklearn Wine dataset")
        data = load_wine()
        X, y = data.data, data.target

    return train_test_split(X, y, test_size=0.2, random_state=42)

def compare_estimators(X_train, X_test, y_train, y_test):
    print("\nComparing Number of Trees (Estimators)")
    n_trees_list = [1, 5, 10, 20, 50]
    my_acc = []
    sk_acc = []

    for n in n_trees_list:
        print(f"Training with n_trees={n}")

        # Our Implementation
        my_model = RandomForest(n_trees=n, max_depth=10)
        my_model.fit(X_train, y_train)
        my_acc.append(accuracy_score(y_test, my_model.predict(X_test)))

        # Scikit-Learn
        sk_model = RandomForestClassifier(n_estimators=n, max_depth=10, random_state=42)
        sk_model.fit(X_train, y_train)
        sk_acc.append(accuracy_score(y_test, sk_model.predict(X_test)))

    return n_trees_list, my_acc, sk_acc

def compare_depths(X_train, X_test, y_train, y_test):
    print("\nComparing Max Depth")
    depths = [2, 5, 10, 20]
    my_acc = []
    sk_acc = []

    for d in depths:
        print(f"Training with max_depth={d}")

        # Our Implementation (fixed trees=10 for speed)
        my_model = RandomForest(n_trees=10, max_depth=d)
        my_model.fit(X_train, y_train)
        my_acc.append(accuracy_score(y_test, my_model.predict(X_test)))

        # Scikit-Learn
        sk_model = RandomForestClassifier(n_estimators=10, max_depth=d, random_state=42)
        sk_model.fit(X_train, y_train)
        sk_acc.append(accuracy_score(y_test, sk_model.predict(X_test)))

    return depths, my_acc, sk_acc

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()

    est_x, est_my, est_sk = compare_estimators(X_train, X_test, y_train, y_test)

    dep_x, dep_my, dep_sk = compare_depths(X_train, X_test, y_train, y_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(est_x, est_my, 'o-', label='My Implementation')
    ax1.plot(est_x, est_sk, 's--', label='Scikit-Learn')
    ax1.set_title('Accuracy vs Number of Trees')
    ax1.set_xlabel('Number of Trees')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(dep_x, dep_my, 'o-', label='My Implementation')
    ax2.plot(dep_x, dep_sk, 's--', label='Scikit-Learn')
    ax2.set_title('Accuracy vs Max Depth')
    ax2.set_xlabel('Max Depth')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    output_file = "comparision_plots.png"
    plt.savefig(output_file)
    print(f"\nCharts saved to file: {output_file}")
    plt.show()
