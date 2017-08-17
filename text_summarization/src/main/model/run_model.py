#!usr/bin/env/python

import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from main.resources import Loading
from main.model.seq2seq import Seq2Seq

sys.stderr.write('Package Using TensorFlow Version: {} \n '.format(tf.__version__))

def main(configuration_file, output_directory, type):

    #output directory should be the same directory that contains the processed data!!!
    assert (os.path.isdir(output_directory) == True)

    configuration = Loading.read_json(configuration_file)['model']

    try:
        processed_data = Loading.load_pickle(os.path.join(output_directory, configuration['processed_data']))
        vocab_to_int = Loading.load_pickle(os.path.join(output_directory, configuration['word_to_ind']))
        int_to_vocab = Loading.load_pickle(os.path.join(output_directory, configuration['ind_to_word']))
        word_embedding_matrix = Loading.load_pickle(os.path.join(output_directory, configuration['embed_matrix']))

    except OSError:
        print('One of the files is missing')

    keys = ['content', 'title']
    sorted_texts = [v for d in processed_data for k, v in d.items() if k == keys[0]]
    sorted_summaries = [v for d in processed_data for k, v in d.items() if k == keys[1]]

    #intitial split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sorted_texts, sorted_summaries,
                                                        test_size=0.20, random_state=11)

    tf.reset_default_graph()

    if type == 'train':

        with tf.Session() as sess:
            model = Seq2Seq(configuration, word_embedding_matrix, vocab_to_int, int_to_vocab, 'training')
            model.build()
            data = (X_train, y_train)
            loss_history = model.train(sess, data, from_scratch=True,
                                       load_ckpt= model.checkpoint,
                                       save_path=model.checkpoint)


    elif type == 'predict':

        with tf.Session() as sess:
            model = Seq2Seq(configuration, word_embedding_matrix, vocab_to_int, int_to_vocab, 'inference')
            model.build()
            data = (X_test, y_test)
            loss_history = model.inference(sess, data, configuration['checkpoint_directory'])