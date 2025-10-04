from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
import pandas as pd
from get_features import get_dataframes, get_labels
import os
from sklearn.ensemble import RandomForestClassifier


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


def get_acc(X_train, y_train, X_test, y_test, list_nodes, classifierName):
    """
    loops through different numbers of max leaf nodes and fits a decision tree classifier
    and records the train accuracy and test accuracy.

    Args:
        X_train (array): a 2-dimentional array that stores the samples for train features. shape = (num_samples, num_features)
        y_train (list): a 1-dimentional list that stores the samples for train labels. shape = (num_samples)
        X_test (array): a 2-dimentional array that stores the samples for test features. shape = (num_samples, num_features)
        y_test (list): a 1-dimentional list that stores the samples for test labels. shape = (num_samples)
        list_nodes (list): a list of the nodes we are testing with.
        classifierName (String): A string indicating the type of classifier to use

    Returns:
        train_accs (list): a list of the train scores from each node.
        test_accs (list): a list of the test scores from each node.
    """
    train_accs = []
    test_accs = []

    for node in list_nodes:
        if classifierName == "DecisionTree":
            clf = DecisionTreeClassifier(max_leaf_nodes=node, random_state=0)
        elif classifierName == "RandomForest":
            clf = RandomForestClassifier(max_leaf_nodes=node, random_state=0)
        else:
            raise NotImplementedError
        clf.fit(X_train, y_train)
        train_accs.append(clf.score(X_train, y_train))
        test_accs.append(clf.score(X_test, y_test))
        

    return train_accs, test_accs

def decisionTreeAnalysis(X, Y, labelNames, classifierName, max_nodes = 24):
    """ graphs a comparison of train and test accuracy using labels provided and plots a confusion
    matrix of the train and test with 12 and 24 nodes.

    Args:
        X (dataframe): features we are using to predict
        Y (dataframe): labels we are trying to predict 
        labelNames (string): uses it to name the files
        classifierName (String): determines which chlassifier to use
        max_nodes (int, optional): amount of nodes to use. Defaults to 24.

    Raises:
        NotImplementedError: _description_
    """
    # place to put results
    results_dir = f"{os.pardir}/results/"
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42)
    plot_acc(X, Y, labelNames, classifierName, max_nodes = 24)

    # determines which classifier to use
    if classifierName == "DecisionTree":
        clf = DecisionTreeClassifier(max_leaf_nodes=max_nodes, random_state=0)
    elif classifierName == "RandomForest":
        clf = RandomForestClassifier(max_leaf_nodes=max_nodes, random_state=0)
    else:
        raise NotImplementedError
    
    #makes a confusion matrix on 12 nodes train and test
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_train, y_pred)

    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax  = plt.subplots(nrows=2, ncols=2, sharex = True, sharey=True)
    ax[0,0].set_title('12 node on train')
    ax[0,1].set_title("12 node on test")
    ConfusionMatrixDisplay.from_estimator(clf, X_train, y_train, ax = ax[0,0])
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax= ax[0, 1])

    # makes confusion matrix on 24 nodes on train and test

    clf2 = DecisionTreeClassifier(max_leaf_nodes=24, random_state=0)
    clf2.fit(X_train, y_train)

    y_pred = clf2.predict(X_train)
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_train, y_pred)

    ax[1,0].set_title('24 node on train')
    ax[1,1].set_title("24 node on test")
    ConfusionMatrixDisplay.from_estimator(clf2, X_train, y_train, ax = ax[1, 0])
    ConfusionMatrixDisplay.from_estimator(clf2, X_test, y_test, ax= ax[1, 1])

    fig.suptitle(f"comparing 12 node and 24 node decision trees \n on ephys data classifier: {classifierName}")
    plt.tight_layout()

    # saves plot as a png file in results
    plt.savefig(f"{results_dir}comparingNodes_ephys{labelNames}_classifier_{classifierName}.png")

def plot_acc(X, Y, labelNames, classifierNames, max_nodes = 24):
    """plots train and test accuracy for the different classifiers used

    Args:
        X (dataframe): features we are using to predict
        Y (dataframe): labels we are trying to predict 
        labelNames (string): uses it to name the files
        classifierNames (list): a list of the classifiers used
        max_nodes (int, optional): the number of nodes used. Defaults to 24.
    """
    # spliting data
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42)
    print(f"aa {len(X_train)}")

    list_nodes = np.arange(2, max_nodes)

    # get accuracy for train and test
    train_dict = {}
    test_dict = {}
    
    for classifier in classifierNames:
        train_acc, test_acc = get_acc(X_train, y_train, X_test, y_test, list_nodes, classifier)
        train_dict[classifier] = train_acc
        test_dict[classifier] = test_acc
    print(train_dict, test_dict)
    fig, ax  = plt.subplots()
    # plots a graph comparing train and test on accuracy
    colors = ["red", "blue", "green", "purple"]
    for ind, classifier in enumerate(train_dict.keys()):
        plt.plot(list_nodes, train_dict[classifier], label = f"{classifier} train", color = colors[ind], linestyle = "solid")
    for ind, classifier in enumerate(test_dict.keys()):
        plt.plot(list_nodes, test_dict[classifier], label = f"{classifier} test", color = colors[ind], linestyle = "--")
    
    plt.xlabel("number of nodes")
    plt.ylabel("accuracy")
    plt.title("decision tree performance on ephys data")
    plt.legend()
    plt.show()
    results_dir = f"{os.pardir}/results/"
    classifierName = "_".join(classifierNames)
    plt.savefig(f"{results_dir}DecisionTreePerformance{labelNames}_classifier_{classifierName}_ephys.png")

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

