#!usr/bin/env/python

import string
import enchant
import re, random
import numpy as np
from collections import Counter
from collections import defaultdict
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from main.resources.english_contractions import Contractions
from unidecode import unidecode
from textblob import Word

class PreProcessing(object):

    def __init__(self, configuration):
        self.code = configuration['code']
        self.language = configuration['language']
        self.number_to_collect = configuration['sample.sentence']
        self.sample_sentence = configuration['sample.sentence']
        self.stop_words = configuration['stop.words']
        self.expand_words = configuration['expand.words']
        self.part_of_speech = configuration['part.of.speech']
        self.remove_keys = configuration['remove.keys']
        self.dictionary = enchant.Dict("en_US")

    def tokenize_sentence(self, text):
        return sent_tokenize(text)

    def remove_non_ascii(self, text):
        return unidecode(text, encoding = "utf-8")

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

        """
        Take the first three sentences of the paragraph
        Do we want to randomly sample sentences from the paragraph (if number of sentences is >3)
        If random sample = True, then we randomly sample 2 sentences from the rest of the paragraph
        """
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
        translation = str.maketrans("","", string.punctuation)
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
            updated_text.append(Contractions[word]) if word in Contractions else updated_text.append(word)
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
        return ' '.join(text)

    def regex_removal(self,text):

        #remove edge cases
        text = re.sub(r'[_"\-;”»–%()|+<>&=*%.,!?:#$@\[\]/]', ' ', text) #remove symbols from text
        text = re.sub(r'\n', '', text) #remove new line "\n"
        text = re.sub(r'--', '', text) # remove "--"
        text = re.sub(r'\\', '', text) # remove "\\"
        text = re.sub(r'< eos >', '<eos>', text) # remove whitespace
        text = re.sub(r'``', '', text) #remove
        text = re.sub(r'avg', 'average', text) #replace
        text = re.sub(r'dbl', 'double', text) #replace
        text = re.sub(r'dist', 'distance', text) #replace
        text = re.sub(r'hwy', 'highway', text) #replace
        text = re.sub(r'misc', 'miscellaneous', text) #replace
        text = re.sub(r'pkg', 'package', text) #replace
        text = re.sub(r'rte', 'route', text) #replace
        text = re.sub(r'wpm', 'words_per_minute', text) #replace
        text = re.sub(r'yr', 'year', text) #replace
        text = re.sub(r'u', 'you', text) #replace
        text = re.sub(r'st', 'street', text) #replace
        text = re.sub(r'sept', 'september', text) #replace
        text = re.sub(r'sep', 'september', text) #replace
        text = re.sub(r'pr', 'public_relations', text) #replace
        text = re.sub(r'co', 'company', text) #replace
        text = re.sub(r'services', 'service', text) #replace
        text = re.sub(r'york', 'new_york', text) #replace
        text = re.sub(r'uk', 'united_kingdom', text) #replace
        text = re.sub(r'san', 'san_francisco', text) #replace
        text = re.sub(r'you', 'u', text)  # replace
        #text = re.sub(r'u', 'you', text) #replace
        #text = re.sub(r'\'s', '', text) #replace
        #text = re.sub(r'(z|v|w|x|xi|xv|xvi)', '', text) #replace
        return text

    def cleaning_text(self, k, text):
        text = self.tokenize_sentence(text)
        if self.sample_sentence:
            text = self.sample_sentence_method(text)
        if self.part_of_speech == 'True' and k == 'content':
            tags = self.part_of_speech_tagging(text)
        text = self.end_of_sentence(text)
        text = self.lower_case(text)
        if self.expand_words:
            text = self.expand_contractions(text)
        text = self.lemmatization(text)  #TODO: move lemmatization step after stop words
        text = self.in_dictionary(text)
        if self.stop_words:
            text = self.remove_stop_words(text)
        text = self.regex_removal(text)
        if self.part_of_speech == 'True' and k == 'content':
            text = '{} {}'.format(text, tags)
        return text

class PostProcessing(object):

    def __init__(self, configuration):
        self.code = configuration['code']
        self.threshold_count = configuration['threshold.count']
        self.embed_dim = configuration['embed.dim']
        self.max_length = configuration['max.length']
        self.embedding_data = configuration['embedding.data']

    #distribution of words in the corpus
    def count_words(self,processed_li):
        counter_dictionary = defaultdict(int)
        for sample in processed_li:
            for v in sample.values():
                text = word_tokenize(v)
                for word in text:
                    counter_dictionary[word] += 1 if word in counter_dictionary else 1
        return counter_dictionary

    def remove_integer_key(self, dictionary):

        """
        :param dictionary: dictionary of all unique tokens in corpora, key=word & value=count_in_corpus
        :return: removes keys from the dictionary that are stringified integers
        """
        remove_keys = []
        for key in dictionary.keys():
            try:
                if isinstance(int(key), int):
                    remove_keys.append(key)
            except:
                pass

        for key in remove_keys: del dictionary[key]
        return dictionary

    def remove_day_tokens(self,dictionary):  #Removing tokens with the following format from the vocab: 17th, 72nd, 31st, 23rd
        p = re.compile('^[0-9]{1,3}(th|rd|nd|st)$')
        remove_keys = [key for key in dictionary if p.match(key)]
        for key in remove_keys: del dictionary[key]
        return dictionary

    def get_embeddings(self):
        embedding_dic = {}
        with open(self.embedding_data) as file:
            for line in file:
                vector = line.split()
                word = vector[0]
                embedding = np.array(vector[1:], dtype='float32')
                embedding_dic[word] = embedding
        return embedding_dic

    def apply_threshold_count(self,dictionary):
        return {k: v for k, v in dictionary.items() if v >= self.threshold_count}

    def set_vocabulary_size(self, dictionary):
        if len(dictionary) < self.max_length:
            return dictionary
        set_counter = Counter(dictionary)
        return dict(set_counter.most_common(self.max_length))  # dictionary with the top occuring words

    def prune_and_embed(self, counter_dictionary, embedding_list):

        assert(self.embed_dim > 100)

        corpora_count = self.remove_day_tokens(self.remove_integer_key(counter_dictionary))
        corpora_count_threshold = self.apply_threshold_count(corpora_count) #set threshold, if greater than threshold, keep
        corpora_count_fixed_size = self.set_vocabulary_size(corpora_count_threshold)  #truncate to maximum vocabulary set by max_length parameter
        vocabulary_words = sorted(list(set(corpora_count_fixed_size.keys())))

        word_to_ind = dict((c, i) for i, c in enumerate(vocabulary_words))
        ind_to_word = dict((i, c) for i, c in enumerate(vocabulary_words))

        #add codes into dictionary (both char -> int and int -> char)
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
                #TODO: deal with mispelled words that can't be found in the word embeddings, maybe take a look at synsets
                create_embedding = np.array(np.random.uniform(-1.0, 1.0, self.embed_dim))
                embed_matrix[index] = create_embedding
                words_without_embeddings.append(word)

        return word_to_ind, ind_to_word, embed_matrix, words_without_embeddings

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

    assert(len(v) == min_len)

    return v

def remove_reviews_and_set_padding(processed_li_to_ind, count_dataframe, word_to_ind, threshold):

    # assuming a gaussian distribution, using about 2 standard deviations away from the mean
    content_len = int(np.percentile(count_dataframe.content, 60))
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
