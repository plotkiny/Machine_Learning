#!usr/bin/env/python

import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from main.resources.helper_function import Loading
from main.resources.helper_function import get_join_dir
from main.model.tensorflow.models.seq2seq_attention import AttentionModel
from main.model.tensorflow.models.seq2seq_model import BasicSeq2Seq
from main.resources.helper_function import get_boolean

sys.stderr.write('Package Using TensorFlow Version: {} \n '.format(tf.__version__))

def main(config_file, output_dir, type):

    #output directory should be the same directory that contains the processed data!!!
    assert (os.path.isdir(output_dir) == True)

    params = Loading.read_json(config_file)["model"]
    processed_data = Loading.load_pickle(get_join_dir(output_dir, params["processed_data"]))
    keys = ["content", "title"]
    sorted_texts, sorted_summaries = AttentionModel._get_text(keys, processed_data)

    #intitial split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sorted_texts, sorted_summaries,
                                                        test_size=float(params['train_test_split']), random_state=params['random_state'])

    tf.reset_default_graph()

    with tf.Session() as sess:
        if get_boolean(params["use_attn"]):
            model = AttentionModel(params, type, output_dir)
        else:
            model = BasicSeq2Seq(params, type, output_dir)

        model._build()

        if type == "train":
                data = (X_train, y_train)
                loss_history = model.train(sess, data, from_scratch=True, load_ckpt= model.checkpoint,
                                           save_path=model.checkpoint)
        elif type == "predict":
                data = (X_test, y_test)
                loss_history = model.inference(sess, data, params["checkpoint_dir"])
        else:
            ValueError("Please specify whether to 'train' or 'predict'")