#!usr/bin/env/python

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
