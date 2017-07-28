

import sys,json, re
import numpy as np
import pandas as pd
import pickle
import enchant
from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

sys.path.append("/home/paperspace/Documents/code/pipeline2/")
from english_contractions import contractions

#downstream dependencies
codes = ['<unk>','<pad>','<eos>','<go>']
english_dictionary = enchant.Dict("en_US")nglish_contractions import contractions

#check if json file or csv
li=[]
with open("/home/paperspace/Documents/data/signalmedia-1m.jsonl") as file:
    for line in file:
        j = json.loads(line)
        li.append(j)

def text_cleaner(text, number_of_sentences = 5, remove_stop_words=True, expand_words=True):
    
    #keep the first few => returns a list of sentences
    text = sent_tokenize(text)
    if len(text) > 5:
        text = text[:number_of_sentences]
        
    #add end of sentence tag "<EOS>"
    text  = ['{} {}'.format(sentence,codes[2]) for sentence in text]
        
    #lower case => returns a single list of words (i.e. ['today was a good day the dog ate my homework'])
    text = [x.lower() for x in text]
    text = ' '.join(text)
    
    #replace contraction words with the longer form
    if expand_words:
        updated_text = []  #updated_text is a list of words
        for word in text.split():
            updated_text.append(contractions[word]) if word in contractions else updated_text.append(word)

    #lemmatization step
    lemma = WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in updated_text]
    
    #check if the word is in the english dictionary
    text = [word for word in text if english_dictionary.check(word)]
    text = ' '.join(updated_text)  
    
    #remove stop words
    #TODO: KEEP OR REMOVE STOP WORDS
    if remove_stop_words:
        text = word_tokenize(text)
        stops = set(stopwords.words("english"))
        text = [word for word in text if not word in stops]
        text = ' '.join(text)
        
    #remove edge cases
    text = re.sub(r'[_"\-;”»–%()|+&=*%.,!?:#$@\[\]/]', ' ', text) #remove symbols from text
    text = re.sub(r'\n', '', text) #remove new line "\n"
    text = re.sub(r'--', '', text) # remove "--"
    text = re.sub(r'\\', '', text) # remove "\\"
    text = re.sub(r'< eos >', '<eos>', text) # remove whitespace
    text = re.sub(r'``', '', text) #remove ``
    
    
    # look into adapting '6', 'p', 'm'
    # look into removing ''
    #text = re.sub(r'[0-9]{2}/[09]{2}/[0-9]{2}', '', text) #replace date 09/17/15 
    
    #!!!!!!!!!!!!! remove url http://www2.marketwire.com/mw/frame_mw?attachid=2889222 
    # deal with a .com like example.com
    # 43 7311 49 3792  numbers only but do not remove 100 mph or age -> 31 
    # 'idinl1n11l1f720150918' token???
    # be cognizant of 2016 year as token, and not random numbers from above
    # 請各位幫我看看
    # 
    
    return text
        
        
#apply text_cleaner to the entire dataset
remove_keys = ['id','published'] 
processed_li = []

for sample in li:
    
    #remove unneeded key:value items
    #what the computational load of a creating a new dictionary vs. popping the key out of the dictionary???
    text_dictionary = {k:v for k,v in sample.items() if k not in remove_keys}
    
    #get the content, title and apply pre-processing steps to them
    #TODO: can apply iterative structure to avoid repeat (for applying the function)
    text_dictionary['content'] = text_cleaner(text_dictionary['content'])
    text_dictionary['title'] = text_cleaner(text_dictionary['title'])
    text_dictionary['media-type'] = text_cleaner(text_dictionary['media-type'])
    text_dictionary['source'] = text_cleaner(text_dictionary['source'])
    processed_li.append(text_dictionary)

#distribution of words in the corpus
counter_dictionary = {}
for sample in processed_li:
    for v in sample.values():
        text = word_tokenize(v)
        for word in text:
            if word not in counter_dictionary:
                counter_dictionary[word] = 1
            else:
                counter_dictionary[word] += 1

set_counter = Counter(counter_dictionary)

#get embeddings list using ConceptNet
embedding_list = {}
def get_embeddings():
    
    with open('/home/paperspace/Documents/data/numberbatch-en-17.04b.txt') as file:
        for line in file:
            vector = line.split()
            word = vector[0]
            embedding = np.array(vector[1:], dtype='float32')
            embedding_list[word] = embedding
#find the words that do not have embeddings from ConceptNet
#Set a threshold count to be used in our vocabulary and remove words that do not meet the count

#TODO: 1-hot encoding vs index features??
#TODO: Determine the right vocabulary size -> add a check in the prune_and_embed method 
#to take the 25K most frequent words after the threshold check. Limiting the size of the vocabulary to somewhere there...

words_without_embeddings = []

def prune_and_embed(set_counter, embedding_list, codes, words_without_embeddings,
                    threshold_count = 1, embed_dim = 300, max_length = 40000):
    
    assert(embed_dim > 100)
    
    #set threshold, if greater than threshold, keep
    corpora_count_with_threshold = {k:v for k,v in set_counter.items() if v >= threshold_count}
    
    #truncate to maximum vocabulary set by max_length parameter
    if len(corpora_count_with_threshold) > max_length:
        set_counter = Counter(corpora_count_with_threshold)
        corpora_count_with_threshold = dict(set_counter.most_common(max_length)) #dictionary with the top occuring words

    vocabulary_words = sorted(list(set(corpora_count_with_threshold.keys())))
    
    word_to_ind = dict((c, i) for i, c in enumerate(vocabulary_words))
    ind_to_word = dict((i, c) for i, c in enumerate(vocabulary_words))
    
    #add codes into dictionary (both char -> int and int -> char)
    for c in codes:
        length = len(word_to_ind)
        word_to_ind[c] = length
        ind_to_word[length] = c
        
    assert(len(word_to_ind) == len(ind_to_word)) #check that they are the same length
    
    #embed variables
    length = len(word_to_ind)
    embed_matrix = np.zeros((length, embed_dim), dtype='float32')
    
    #check if the filtered (via threshold words have embeddings from ConceptNet)
    for word, index in word_to_ind.items():
        if word in embedding_list:
            embed_matrix[index] = embedding_list[word]  #if the word is in ConceptNet
        else:
            create_embedding = np.array(np.random.uniform(-1.0, 1.0, embed_dim))
            embed_matrix[index] = create_embedding
            words_without_embeddings.append(word)
            
    return(word_to_ind,ind_to_word,embed_matrix, words_without_embeddings)


#calling the prune_and_embed_function
#TODO: verify that i even need the "words_without_embeddings" variable in the code base
word_to_ind,ind_to_word,embed_matrix, words_without_embeddings = prune_and_embed(set_counter, embedding_list, 
                                                                                 codes, words_without_embeddings)
            
with open('/home/paperspace/Documents/data/word_to_ind.txt', 'wb') as output:
    pickle.dump(word_to_ind, output)

with open('/home/paperspace/Documents/data/ind_to_word.txt', 'wb') as output:
    pickle.dump(ind_to_word, output)
    

#converting all words to integers
#calculating the distribution of lengths for each field 
processed_li_to_ind = []
columns = ['content','media_type','source','title']
count_dataframe = pd.DataFrame(columns=columns)

def get_index(word, word_to_ind):
    return word_to_ind[word] if word in word_to_ind else word_to_ind["<unk>"]

def convert_to_ind(value):    
    return ' '.join([ str(get_index(x,word_to_ind)) for x in value.split() ])

for sample in processed_li:
    
    len_of_sentence = {}
    
    #sample = dictionary containing the content, title, media-type and source
    for k,v in sample.items():
        
        v = convert_to_ind(v) #convert to integers
        sample[k] = v 
        
        #collect length distribution information, length parameter is the # of tokens
        len_of_sentence[k] = len(v.split())
          
    #add new converted into processed list
    processed_li_to_ind.append(sample)
    
    #count lengths
    len_values = [v for k,v in sorted(len_of_sentence.items())]
    dataframe = pd.DataFrame([len_values], columns=columns)
    count_dataframe = count_dataframe.append(dataframe)

print(count_dataframe.describe())
print(int(np.percentile(count_dataframe.content, 96)));
print(np.percentile(count_dataframe.media_type, 96));
print(np.percentile(count_dataframe.source, 96));
print(np.percentile(count_dataframe.title, 96));



#remove reviews if they have too many unknown (as a result of filtering scarce words from threshold set upstream)
#TODO: from a sample of 1000, the average number of words is 244 (99th percentile). Think about better algorithmor implementation

#higher-order function stuff -> decorator
#v = value of string, padding_index = word_to_ind['<pad>'], min_len = minimum key_specific length

def padding_function(f):
    def wrapper(v, min_len):
        return f(v, min_len)
    return wrapper

@padding_function
def is_padding_required(v, min_len):
    
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

def remove_reviews_and_set_padding(processed_li_to_ind, threshold = .15):
    
    #assuming a gaussian distribution, using about 2 standard deviations away from the mean
    content_len = int(np.percentile(count_dataframe.content, 96)) 
    media_type_len = int(np.percentile(count_dataframe.media_type, 96)) 
    source_len = int(np.percentile(count_dataframe.source, 96)) 
    title_len = int(np.percentile(count_dataframe.title, 96)) 
    
    #removing samples with too many unknown words
    #index <unk> = 18093
    unknown_index = word_to_ind['<unk>']
    unknown_threshold = int(content_len*threshold)
    
    for sample in processed_li_to_ind:
        for k,v in sample.items():
        
            v = v.split()
            
            #filtering reviews
            if k == 'content':
                
                #remove the sample
                count = int(v.count(unknown_index))
                if count > threshold:
                    processed_li_to_ind.pop(sample)
                    
                #padding, good place to add a higher order function
                sample[k] = is_padding_required(v, content_len)
                    
            elif k == 'media_type':
                sample[k] = is_padding_required(v, media_type_len)
            elif k == 'source':
                sample[k] = is_padding_required(v, source_len)
            elif k == 'title':
                sample[k] = is_padding_required(v, title_len)
                        
    return processed_li_to_ind
            

data = remove_reviews_and_set_padding(processed_li_to_ind, threshold = .15) 
processed_data = []

#concatenate content, source and media-type values (appending to content) into 1 vector
for sample in data:
    
    title = sample['title']
    content = sample['content']; content.extend(sample['source']); content.extend(sample['media-type'].split())
    new_sample = {'title':title, 'content':content}
    processed_data.append(new_sample)

with open('processed_data.txt', 'wb') as output:
    pickle.dump(processed_data, output)


