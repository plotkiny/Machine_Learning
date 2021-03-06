#!usr/bin/env/python

import os
import sys
import pandas as pd
from main.resources.helper_function import Loading
from main.data import nlp


def main(configuration_file, output_directory):

    assert(os.path.isdir(output_directory) == True)

    configuration = Loading.read_json(configuration_file)['pipeline']
    data = Loading.read_data(configuration['input.data'])[730000:738000]

    pre_processing = nlp.PreProcessing(configuration)
    post_processing = nlp.PostProcessing(configuration)

    processed_li = []
    for sample in data:
        text_dictionary = {k:v for k,v in sample.items() if k not in configuration['remove.keys']}   #remove unneeded key:value items
        for k,v in text_dictionary.items():
            text = pre_processing.cleaning_text(k,v)
            text_dictionary[k] = text
        processed_li.append(text_dictionary)

    counter_dictionary = post_processing.count_words(processed_li)
    embedding_list = post_processing.get_embeddings()
    word_to_ind, ind_to_word, embed_matrix, words_without_embeddings = post_processing.prune_and_embed(counter_dictionary,
                                                                                                   embedding_list)

    assert (embed_matrix.shape == (len(word_to_ind), configuration['embed.dim']))

    Loading.save_pickle(os.path.join(output_directory, configuration['word.frequency']), counter_dictionary)
    Loading.save_pickle(os.path.join(output_directory, 'words_without_embedding'), words_without_embeddings)
    Loading.save_pickle(os.path.join(output_directory, configuration['embed.matrix']), embed_matrix)
    Loading.save_pickle(os.path.join(output_directory, configuration['word.to.ind']), word_to_ind)
    Loading.save_pickle(os.path.join(output_directory, configuration['ind.to.word']), ind_to_word)

    #converting all words to integers
    #calculating the distribution of lengths for each field

    columns = ['content','media_type','source','title']
    count_dataframe = pd.DataFrame(columns=columns)

    processed_li_to_ind = []
    for sample in processed_li:
        len_of_sentence = {}
        for k,v in sample.items():  #sample = dictionary containing the content, title, media-type and source
            v = post_processing.convert_to_ind(v, word_to_ind) #convert to integers
            sample[k] = v
            len_of_sentence[k] = len(v.split()) #collect length distribution information, length parameter is the # of tokens
        processed_li_to_ind.append(sample) #add new converted into processed list
        len_values = [v for k,v in sorted(len_of_sentence.items())]  #count lengths
        dataframe = pd.DataFrame([len_values], columns=columns)
        count_dataframe = count_dataframe.append(dataframe)

    # concatenate content, source and media-type values (appending to content) into 1 vector
    processed_data = []
    for sample in processed_li_to_ind:
        title = sample['title']; content = sample['content']
        media = sample['media-type']; source = sample['source']

        content = ' '.join([content, media, source])
        new_sample = {'title': title, 'content': content}
        processed_data.append(new_sample)

    processed_data_with_padding = nlp.remove_reviews_and_set_padding(processed_data, count_dataframe, word_to_ind, threshold = .10)
    Loading.save_pickle(os.path.join(output_directory, configuration['processed.data']), processed_data_with_padding)
    sys.stderr.write('The processed data was saved to {} \n '.format(output_directory))

