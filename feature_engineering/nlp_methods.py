
import spacy
from feature_engineering.loading_data import saveData


def wordPunctuationParser(token):

    """
    Method for tokens: eliminate pure tokens that are either punctuation or whitespace
    """

    return token.is_punct or token.is_space


def sentencePunctuationParser(sent, join=False):

    """
    Method for sentences: eliminate pure tokens that are either punctuation or whitespace
    Input: String (can be 1 or many sentences)
    Output: String with punctuations and whitespace removed from the string
    """

    li = [token.lemma_ for token in sent if not any([token.is_punct, token.is_space])]
    if not join:
        return li
    return ' '.join(li)

def sentenceStopWordsRemoval(sent, join=False):

    """
    Eliminate stop words
    """

    li = [token for token in sent if token not in spacy.en.language_data.STOP_WORDS]
    if not join:
        return li
    return ' '.join(li)


def phraseModelingSentenceGenerator(df, file_path):

    """
    -Purpose: Building phrase models of text --> generate 1 sentence per line (reviews have multiple sentences)
    -Input: Pandas dataframe and a full path to save the data
    -Output: Text file containing one sentence per line.
    -Helper function to perform 2 main operations:
        1. normalize and remove tokens punctuation tokens (see punctuaction_parser method).
        2. Each row in the dataframe is of type 'spacy.tokens.doc.Doc' which contains at least 1 sentence (or more).
           The function get passed r.sents (a generator) and iterates over each sentence, yielding a sentence per call.
    """

    def normalize_text_generator_helper(r):
        for sent in r:
            yield sentencePunctuationParser(sent)

    for review in df:
        for sent in normalize_text_generator_helper(review.sents):
            saveData(sent, file_path)

