from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import sys
import os

# Import helper functions
import pandas as pd
from utils import train_test_split, standardize, accuracy_score
from utils import mean_squared_error, calculate_variance, Plot
from decision_tree_model import ClassificationTree


def create_target(row):
    if row.Name == 'setosa':
        return 0
    elif row.Name == 'versicolor':
        return 1
    else:
        return 2


def main():
    df = pd.read_csv('fishiris.csv')
    df['target'] = df.apply(create_target, axis=1)
    y = df['target'].to_numpy()
    df = df.drop(['Name', 'target'], axis=1)
    feature_names = df.columns.tolist()
    X = df.to_numpy()
    target_names = ['setosa', 'versicolor', 'virginica']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)
    print('X_train\n', X_train)
    print('y_train\n', y_train)
    print('X_test\n', X_test)
    print('y_test\n', y_test)
    clf = ClassificationTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('-' * 40, 'print_tree', '-' * 40)
    clf.print_tree(feature_names=feature_names)
    print('-' * 40, 'print_tree', '-' * 40)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="Decision Tree", accuracy=accuracy, legend_labels=target_names)
    Plot().plot_in_3d(X_test, y_pred)


if __name__ == "__main__":
    main()
