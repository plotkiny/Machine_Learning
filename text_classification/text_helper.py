#!usr/bin/env python

import random
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Feature_Engineering.nlp_methods import nlp_MethodsBase
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

class exploratory():

    @staticmethod
    def describe(df):
        print(df.isnull().sum())
        print(df.describe())

    @staticmethod
    def plot_class_balance(df, label):

        if label == "index":
            x = y = pd.Series(df.index)
        else:
            x = y = df[label]

        fig = plt.figure(figsize=(8, 4))
        sns.barplot(x=x.unique(), y=y.value_counts())
        plt.show()

    @staticmethod
    def plot_most_common_words():
        pass


class nlp_methods(nlp_MethodsBase):

    @classmethod
    def filter_length(self, li):
        return [x for x in li if len(x) > 1 and isinstance(x, str)]

    @classmethod
    def preprocess_text(self, data, model):

        # TODO:need to deal with acronyms/abbreviations: i.e. = w.c,
        # TODO:phrase_modeling i.e. new york --> new_york
        # TODO: convert year to string i.e 1977 --> ninteen_seventy_seven
        # TODO: dealing with named entities and POS

        df = data.copy()

        # remove punctuation, whitespace and stopwords from each review, using spacy objects
        df["text"] = df.apply(lambda x: nlp_MethodsBase.sentenceStopWordsRemoval(nlp_MethodsBase.sentencePunctuationParser(model(x[0]))), axis=1)

        # regex rules
        regex1 = re.compile(r"'s")
        regex2 = re.compile(r"``")
        regex3 = re.compile(r"-PRON-")

        df["text"] = df.apply(lambda x: list(filter(lambda i: not regex1.search(i), x[0])), axis=1)
        df["text"] = df.apply(lambda x: list(filter(lambda i: not regex2.search(i), x[0])), axis=1)
        df["text"] = df.apply(lambda x: list(filter(lambda i: not regex3.search(i), x[0])), axis=1)
        df["text"] = df["text"].apply(nlp_methods.filter_length)  # filter all strings of length == 1
        df = df[df.apply(lambda x: len(x[0]) > 3, axis=1)]
        df["text"] = df.apply(lambda x: " ".join(x[0]), axis=1)

        df["class1"] = df.apply(lambda x: x[1].split("_")[0], axis=1)
        df["class2"] = df.apply(lambda x: x[1].split("_")[1], axis=1)
        return df


def acquire_data(data_file):

    li = []
    with open(data_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().replace(":", "_").replace("\n", " ").lower().split()
            tup = (" ".join(line[1:]), line[0])  # (text, class_label)
            li.append(tup)

    df = pd.DataFrame(li)
    df.columns = ["text", "class"]
    df.drop_duplicates(inplace=True)  # remove duplicate training entries
    print("Before preprocessing shape {}".format(df.shape))
    return df


def load_data(train_data, class_dict, limit=0, split=0.8):
    # Partition off part of the train data for evaluation
    d = dict.fromkeys(class_dict, 0)
    reverse_d = dict([(v,k) for k,v in class_dict.items()])
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = []
    for ind in range(len(labels)):
        d_copy = d.copy()
        key = reverse_d[labels[ind]]
        d_copy[key] = 1
        cats.append(d_copy)
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def generative_model(df, col, label):
    bow_transformer = CountVectorizer()
    bow_transformer.fit(df[col])
    messages_bow = bow_transformer.transform(df[col])

    print("Shape of Sparse Matrix: ", messages_bow.shape)
    print("Amount of Non-Zero occurences: ", messages_bow.nnz)
    print("sparsity: %.2f%%" % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))

    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)

    spam_detect_model = MultinomialNB().fit(messages_tfidf, df[label])
    all_predictions = spam_detect_model.predict(messages_tfidf)

    print("Prediction of Class1 using a generative model")
    print(classification_report(df[label], all_predictions))
    return all_predictions


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}

