#!usr/bin/env/python

import warnings
from sklearn.model_selection import StratifiedShuffleSplit

def stratifiedSplit(df, response_var, n_splits=2, test_size=.15, random_state=22):

    """
    Splitting data into training and test sets.
    """

    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(df, response_var):
        X_train, X_test = df.loc[train_index], df.loc[test_index]
        y_train, y_test = response_var[train_index], response_var[test_index]

    return X_train, y_train, X_test, y_test