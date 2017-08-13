#!usr/bin/env/python

import sys,re, string
import numpy as np
import random
import enchant
from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

sys.path.append("/Users/yuriplotkin/Documents/Paperspace/local_testing/pipeline1/")
from english_contractions import contractions


class PreProcessing(object):

    def __init__(self, configuration):
        self.code = configuration['code']
        self.language = configuration['language']
        self.number_to_collect = configuration['sample_sentence']
        self.sample_sentence = configuration['sample_sentence']
        self.stop_words = configuration['stop_words']
        self.expand_words = configuration['expand_words']
        self.part_of_speech = configuration['part_of_speech']
        self.remove_keys = configuration['remove_keys']
        self.dictionary = enchant.Dict("en_US")

    def tokenize_sentence(self, text):
        return sent_tokenize(text)

    def sample_algorithm(self, text, count):

        #taking the first 3 sentences and then randomly sampling the rest of the paragraph for sentences to use
        #the numner of randomly picked samples is count
        sampled_text = text

        list_of_index = [i for i,j in enumerate(text) if i not in [0,1,2]]
        rndm_index = random.sample(list_of_index, count)
        sampled_text = text[:3]  #take the first three sentences
        for i in rndm_index:
            sampled_text.append(text[i])

        return sampled_text

    def sample_sentence_method(self, text):

        max_length = len(text)
        count = self.number_to_collect - 3

            #take the first three sentences of the paragraph
        #do we want to randomly sample sentences from the paragraph
        #need to make sure the maximum # of sentences is not smaller then number_to_collect
        if not self.sample_sentence:
            index = self.number_to_collect if self.number_to_collect < max_length else max_length
            text = text[:index]
        else:
            if max_length > self.number_to_collect: # m=11, c=10
                text = self.sample_algorithm(text,count) #randomly sample sentences from the paragraph
            else:
                text = text[:max_length]

        return text

    def part_of_speech_tagging(self, text):

        translation = str.maketrans("","", string.punctuation);
        tagged_tuples = [pos_tag(word_tokenize(sentence)) for sentence in text] #list of lists with each list containing (word, tag)
        list_of_tags = [[item[1] for item in li] for li in tagged_tuples] #get the tag for each word, returns list of lists
        list_of_tags = [' '.join(li).translate(translation) for li in list_of_tags] #list of strings with tags
        list_of_tags = [li + '<eos>' for li in list_of_tags] #add end of sentence tag to the end of each string
        list_of_tags = ' '.join(list_of_tags).split() #ouptut is a list of strings --> join into one sentence and split on whitespace (getting rid of double white space)
        list_of_tags = ' '.join(list_of_tags)

        return list_of_tags

    def end_of_sentence(self, text):
        return ['{} {}'.format(sentence, self.code[2]) for sentence in text]

    def lower_case(self,text):
        return ' '.join([x.lower() for x in text])

    def expand_contractions(self,text):
        updated_text = []  #updated_text is a list of words
        for word in text.split():
            updated_text.append(contractions[word]) if word in contractions else updated_text.append(word)
        return updated_text

    def lemmatization(self,text):
        lemma = WordNetLemmatizer()
        text = [lemma.lemmatize(word) for word in text]
        return text

    def in_dictionary(self, text):
        return  ' '.join([word for word in text if self.dictionary.check(word)])

    def remove_stop_words(self,text):
        text = word_tokenize(text)
        stops = set(stopwords.words(self.language))
        text = [word for word in text if not word in stops]
        text = ' '.join(text)
        return text

    def regex_removal(self,text):

        #remove edge cases
        text = re.sub(r'[_"\-;”»–%()|+&=*%.,!?:#$@\[\]/]', ' ', text) #remove symbols from text
        text = re.sub(r'\n', '', text) #remove new line "\n"
        text = re.sub(r'--', '', text) # remove "--"
        text = re.sub(r'\\', '', text) # remove "\\"
        text = re.sub(r'< eos >', '<eos>', text) # remove whitespace
        text = re.sub(r'``', '', text) #remove ``

        return text

    def cleaning_text(self,text):

        text = self.tokenize_sentence(text)

        if self.sample_sentence:
            text = self.sample_sentence_method(text)

        if self.part_of_speech:
            tags = self.part_of_speech_tagging

        text = self.end_of_sentence(text)
        text = self.lower_case(text)

        if self.expand_words:
            text = self.expand_contractions(text)

        text = self.lemmatization(text)
        text = self.in_dictionary(text)

        if self.stop_words:
            text = self.remove_stop_words(text)

        text = self.regex_removal(text)

        if self.part_of_speech:
            text = '{} {}'.format(text, tags)

        return text


class PostProcessing(object):

    def __init__(self, configuration):
        self.code = configuration['code']
        self.threshold_count = configuration['threshold_count']
        self.embed_dim = configuration['embed_dim']
        self.max_length = configuration['max_length']
        self.embedding_data = configuration['embedding_data']

    #distribution of words in the corpus
    def counter(self,processed_li):

        counter_dictionary = {}
        for sample in processed_li:
            for v in sample.values():
                text = word_tokenize(v)
                for word in text:
                    if word not in counter_dictionary:
                        counter_dictionary[word] = 1
                    else:
                        counter_dictionary[word] += 1

        return counter_dictionary

    def get_embeddings(self):

        embedding_dic = {}
        with open(self.embedding_data) as file:
            for line in file:
                vector = line.split()
                word = vector[0]
                embedding = np.array(vector[1:], dtype='float32')
                embedding_dic[word] = embedding

        return embedding_dic

    def prune_and_embed(self, counter_dictionary, embedding_list):

        assert (self.embed_dim > 100)

        # set threshold, if greater than threshold, keep
        corpora_count_with_threshold = {k: v for k, v in counter_dictionary.items() if v >= self.threshold_count}

        # truncate to maximum vocabulary set by max_length parameter
        if len(corpora_count_with_threshold) > self.max_length:
            set_counter = Counter(corpora_count_with_threshold)
            corpora_count_with_threshold = dict(
                set_counter.most_common(self.max_length))  # dictionary with the top occuring words

        vocabulary_words = sorted(list(set(corpora_count_with_threshold.keys())))

        word_to_ind = dict((c, i) for i, c in enumerate(vocabulary_words))
        ind_to_word = dict((i, c) for i, c in enumerate(vocabulary_words))

        # add codes into dictionary (both char -> int and int -> char)
        for c in self.code:
            length = len(word_to_ind)
            word_to_ind[c] = length
            ind_to_word[length] = c

        assert (len(word_to_ind) == len(ind_to_word))  # check that they are the same length

        # embed variables
        length = len(word_to_ind)
        embed_matrix = np.zeros((length, self.embed_dim), dtype='float32')

        # check if the filtered (via threshold words have embeddings from ConceptNet)
        words_without_embeddings = []
        for word, index in word_to_ind.items():
            if word in embedding_list:
                embed_matrix[index] = embedding_list[word]  # if the word is in ConceptNet
            else:
                create_embedding = np.array(np.random.uniform(-1.0, 1.0, self.embed_dim))
                embed_matrix[index] = create_embedding
                words_without_embeddings.append(word)

        return (word_to_ind, ind_to_word, embed_matrix, words_without_embeddings)

    def get_index(self, word, word_to_ind):
        return word_to_ind[word] if word in word_to_ind else word_to_ind["<unk>"]

    def convert_to_ind(self, value, word_to_ind):
        return ' '.join([str(self.get_index(x, word_to_ind)) for x in value.split()])


def padding_function(f):
    def wrapper(v, min_len, word_to_ind):
        return f(v, min_len, word_to_ind)
    return wrapper

@padding_function
def is_padding_required(v, min_len, word_to_ind):
    len_of_v = len(v)
    padding_index = word_to_ind['<pad>']

    if len_of_v > min_len:
        v = v[:min_len]
    elif len_of_v < min_len:
        difference = min_len - len_of_v
        v += difference * [padding_index]
    elif len_of_v == min_len:
        pass

    return v

def remove_reviews_and_set_padding(processed_li_to_ind, count_dataframe, word_to_ind, threshold):

    # assuming a gaussian distribution, using about 2 standard deviations away from the mean
    content_len = int(np.percentile(count_dataframe.content, 60))
    media_type_len = int(np.percentile(count_dataframe.media_type, 60))
    source_len = int(np.percentile(count_dataframe.source, 60))
    title_len = int(np.percentile(count_dataframe.title, 60))

    # removing samples with too many unknown words
    unknown_index = word_to_ind['<unk>']
    unknown_threshold_content = int(content_len * threshold)
    unknown_threshold_title = int(title_len * threshold)

    data = []
    for sample in processed_li_to_ind:

        keep_content = True
        keep_title = True

        for k, v in sample.items():

            v = v.split()
            v = [int(w) for w in v]
            count = int(v.count(unknown_index))

            if k == 'content':
                sample[k] = is_padding_required(v, content_len, word_to_ind)
                if count > unknown_threshold_content:
                    keep_content = False

            elif k == 'title':
                sample[k] = is_padding_required(v, title_len, word_to_ind)
                if count > unknown_threshold_title:
                    keep_title = False

        if all([keep_content, keep_title]):
            data.append(sample)

    return data
