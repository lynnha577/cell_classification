from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from get_features import get_labels

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

def main():
    filtered_dataframe_labels = get_labels()
    Y = filtered_dataframe_labels['dendrite_type_number']
    X = filtered_dataframe_labels[["structure_layer_name_number", "species_number"]]
    max_leaf_nodes = 4
    make_decisionTree(X, Y, max_leaf_nodes)

if __name__ == "__main__":
    main()
