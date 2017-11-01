#!usr/bin/env/python

import json, pickle, os
import pandas as pd
from feature_engineering.cleaning_data import getColumnIndex

def joinPathToFile(file_name, file_path=None):

    """
    Method for joining path to file
    """

    if file_path == None:
        return file_name
    else:
       return os.path.join(file_path,file_name)


def loadCSVToPandas(file, sep=';', time_col = None):

    """
    Method for loading csv data into a pandas dataframe
    """

    try:
        if time_col:
            col = getColumnIndex(time_col)
            return pd.read_csv(file, sep=sep, parse_dates=[col])
        return pd.read_csv(file, sep=sep)
    except IOError:
        print('The file was not found')

def loadJsonToPandas(file):

    """
    Method for loading json data into a pandas dataframe. Each line of the file is a json dictionary.
    """

    try:
        return pd.read_json(file, lines=True)
    except IOError:
        print('The file was not found')

def readData(file):
    li=[]
    with open(file) as f:
        for line in f:
            j = json.loads(line)
            li.append(j)
    return li

def saveData(df,path):
    with open(path, 'a') as f:
        f.write(df + '\n')

def savePickle(file, data):
    with open(file, 'wb') as output:
        pickle.dump(data, output)

def loadPickle(file):
    with open(file, 'rb') as input:
        return pickle.load(input)

