#!usr/bin/env/python

import pandas as pd
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from mlxtend.preprocessing import standardize

def getColumnIndex(df,s):

    """
    Method for retrieving the index of a column. Checks if it's a string and returns index. Otherwise return.
    """

    try:
        if isinstance(s, str):
            return df.columns.get_loc(s)
        elif isinstance(s, int):
            return s
    except TypeError:
        print('Please pass a number as either an integer or string type')


def removeDuplicates(df):

    """
    Remove duplicates from a pandas dataframe. Performs operation inplace.
    """

    print('Removing duplicate observations')
    df.drop_duplicates(inplace=True)


def viewMissingData(df):

    """
    Methods for viewing and dealing with missing data. Missing data can either be removed from the dataframe or imputed
    based on several methods.
    """

    def view_missing_data_helper():
        return lambda y: sum(y.isnull())
    print(df.apply(view_missing_data_helper(), axis=0))


#TODO: NOT COMPATIBLE WITH NUMPY ARRAY, ONLY WORKS WITH PANDAS
def editMissingData(df, interpolate=False, option=0):
    if interpolate:
        if option == 0:
            method = 'barycentric'
        elif option == 1:
            method = 'akima'
        elif option == 2:
            method = 'pchip'
        df.interpolate(method, inplace=True)
    else:
        df.dropna(inplace=True)


def handlingCategoricalOrdinal(df, column):

    """
    Methods for dealing with ordinal  categorical variables. If transforming nominal features in the pandas
    dataframe, it is expected that the column of values is already transformed using the ordinal feature method.
    """

    column = getColumnIndex(df, column) # check if the column is an instance of a string -> convert to integer for indexing
    class_encoder = LabelEncoder()
    df.iloc[:,column] = class_encoder.fit_transform((df.iloc[:, column].values))
    return df


#TODO: need to handle naming of columns before and after transformation
def handlingCategoricalNominal(df, column=None, dummy=False):

    """
    Method for dealing with nominal categorical variables. Dummy encoding converts multiple columns of only type 'str'

    """

    if dummy:
        return pd.get_dummies(df)
    else:
        #for one-hot encoding a single column, output is a numpy array that needs (no column header names)
        column = getColumnIndex(df, column) # check if the column is an instance of a string -> convert to integer for indexing
        one_hot = OneHotEncoder(categorical_features=[column])
        return one_hot.fit_transform(df).toarray()


def getColumns(df, start, stop=None):

    """
    Subsetting columns in a pandas dataframe.
    """

    if not stop:
        stop = df.shape[1]
    stop = getColumnIndex(df,stop)
    start = getColumnIndex(df,start)
    return df.iloc[:, start:stop]


def timeParser(df, col, str_method):

    """
    -method: time_parser
    -description:
        -uses: parse time date in the format 2014-09-18 11:47:45 using any parse method
        -inputs:
            1. name of the pandas dataframe
            2. string of the name of the date column
            3. string of the method you want the parser to get
        -output: pd.Setries the length of the dataframe rows
        -getattr gets the method of the parsed object. the method is specified as an input parameter
    -dependencies: from dateutil.parser import parse
    """

    def time_parser_helper(col, str_method):
        return lambda row: getattr(parse(str(row[col])), str_method)
    return df.apply(time_parser_helper(col,str_method), axis=1)


def standardizeNumpyColumns(X, col_index_list, params=None):

    """
    Method using the mlxtend python package to standardizing select columns in 2-D numpy array
    Input: Numpy array: (rows = samples, columns = features) + col_index_list: list of columns to standardize
    """

    if not params:
        X_std, params =  standardize(X, columns=col_index_list, return_params=True)
        X[:, col_index_list] = X_std
        return X, params
    else:
        X_std =  standardize(X, columns=col_index_list, params=params)
        X[:,col_index_list] = X_std
        return X
