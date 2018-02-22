#!usr/bin/env/python

import os
import json
import pickle

class Loading():

    @staticmethod
    def read_data(file):
        li=[]
        with open(file) as f:
            for line in f:
                j = json.loads(line)
                li.append(j)
        return li

    @staticmethod
    def read_json(file):
        with open(file) as f:
            return json.load(f)

    @staticmethod
    def save_pickle(file, data):
        with open(file, 'wb') as output:
            pickle.dump(data, output)

    @staticmethod
    def load_pickle(file):
        with open(file, 'rb') as input:
            return pickle.load(input)


def get_boolean(bool):
    bool = bool.strip().lower()
    if bool == "true":
        return True
    elif bool == "false":
        return False
    else:
        TypeError("String needs to be of type True or False.")

def check_string(str):
    return None if str.lower() == "none" else str

def get_join_dir(a,b):
    return os.path.join(a,b)
