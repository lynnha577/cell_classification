from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
import pandas as pd
from get_features import get_dataframes, get_labels
import os

def make_decisionTree(X, Y, max_leaf_nodes):
    """
    creates a decision tree and fits to x and y train and plots the tree.

    Args:
        X (dataframe): the inputs to the model.
        Y (dataframe): the labels we want to predict.
        max_leaf_nodes (int): the max number of leaf nodes for the tree.
    """

    X = X.to_numpy()
    Y = Y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42)

    clf = DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes, random_state=0)
    clf.fit(X_train, y_train)

    tree.plot_tree(clf, proportion=True)
    plt.show()


def get_acc(X_train, y_train, X_test, y_test, list_nodes):
    """
    loops through different numbers of max leaf nodes and fits a decision tree classifier
    and records the train accuracy and test accuracy.

    Args:
        X_train (array): a 2-dimentional array that stores the samples for train features. shape = (num_samples, num_features)
        y_train (list): a 1-dimentional list that stores the samples for train labels. shape = (num_samples)
        X_test (array): a 2-dimentional array that stores the samples for test features. shape = (num_samples, num_features)
        y_test (list): a 1-dimentional list that stores the samples for test labels. shape = (num_samples)
        list_nodes (list): a list of the nodes we are testing with.

    Returns:
        train_accs (list): a list of the train scores from each node.
        test_accs (list): a list of the test scores from each node.
    """
    train_accs = []
    test_accs = []

    for node in list_nodes:
        clf = DecisionTreeClassifier(max_leaf_nodes=node, random_state=0)
        clf.fit(X_train, y_train)
        train_accs.append(clf.score(X_train, y_train))
        test_accs.append(clf.score(X_test, y_test))

    return train_accs, test_accs

def decisionTreeAnalysis(X, Y, labelNames, max_nodes = 24):
    results_dir = f"{os.pardir}/results/"
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42)

    list_nodes = np.arange(2, max_nodes, 2)

    train_acc, test_acc = get_acc(X_train, y_train, X_test, y_test, list_nodes)
    print(train_acc)
    print(test_acc)

    fig, ax  = plt.subplots()

    plt.plot(list_nodes, train_acc, label = "train")
    plt.plot(list_nodes, test_acc, label = "test")
    plt.xlabel("number of nodes")
    plt.ylabel("accuracy")
    plt.title("decision tree performance on ephys data")
    plt.legend()
    plt.show

    plt.savefig(f"{results_dir}DecisionTreePerformance{labelNames}_ephys.png")

    #12 nodes
    fig, ax  = plt.subplots(nrows=2, ncols=2, sharex = True, sharey=True)

    clf = DecisionTreeClassifier(max_leaf_nodes=12, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_train, y_pred)
    print(matrix)

    from sklearn.metrics import ConfusionMatrixDisplay
    ax[0,0].set_title('12 node on train')
    ax[0,1].set_title("12 node on test")
    ConfusionMatrixDisplay.from_estimator(clf, X_train, y_train, ax = ax[0,0])
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax= ax[0, 1])

    # 24 nodes 

    clf2 = DecisionTreeClassifier(max_leaf_nodes=24, random_state=0)
    clf2.fit(X_train, y_train)

    y_pred = clf2.predict(X_train)
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_train, y_pred)
    print(matrix)

    ax[1,0].set_title('24 node on train')
    ax[1,1].set_title("24 node on test")
    ConfusionMatrixDisplay.from_estimator(clf2, X_train, y_train, ax = ax[1, 0])
    ConfusionMatrixDisplay.from_estimator(clf2, X_test, y_test, ax= ax[1, 1])

    fig.suptitle("comparing 12 node and 24 node decision trees \n on ephys data")
    plt.tight_layout()

    plt.savefig(f"{results_dir}comparingNodes_ephys{labelNames}.png")


def main():
    labels = get_labels()
    epys_features = get_dataframes()

    full_dataframe = pd.merge(epys_features, labels, left_index=True, right_index=True)


    X = full_dataframe.drop(labels.columns, axis=1)
    Y = full_dataframe["dendrite_type"]
    max_leaf_nodes = 4
    make_decisionTree(X, Y, max_leaf_nodes)
    decisionTreeAnalysis(X, Y)


if __name__ == "__main__":
    main()

