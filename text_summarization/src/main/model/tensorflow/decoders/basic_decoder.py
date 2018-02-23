

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
from main.model.tensorflow.configurable import Configurable
from main.resources.helper_function import get_boolean
from main.resources.helper_function import check_string
from tensorflow.python.layers import core as layers_core

class BasicDecoder(Configurable):

    def __init__(self, params, mode, output_dir, vocab_size, summ_length, max_length, name="basic_decoder"):
        Configurable.__init__(self, params, mode, output_dir)
        self.vocab_size = vocab_size
        self.initializer = check_string(self.params["initialize_dense"])
        self.summ_length = summ_length
        self.max_length = max_length

    def _setup_parameters(self, dec_param_list):
        """Sets decoder parameters """
        self.enc_output = dec_param_list[0]
        self.seq_length = dec_param_list[1]
        self.enc_final_state = dec_param_list[2]
        self.dynamic_batch_size = dec_param_list[3]

    def _create_decoder_object(self, dec_cell, dec_embed_input, dec_init_state, output_layer):
        if self.mode == "predict":
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.embed_matrix,
                start_tokens=self.start_token,
                end_token=self.end_token)
        else:
            helper = tf.contrib.seq2seq.TrainingHelper(  # creating the training logits
                inputs=dec_embed_input,
                sequence_length=self.summ_length,
                time_major=False)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=dec_cell,
            helper=helper,
            initial_state=dec_init_state,
            output_layer=output_layer)

        return {"helper":helper, "decoder":decoder}

    def _setup_decoder(self, initial_state, helper, decoder):
        """Sets the initial state and helper for the decoder.
        """
        self.initial_state = initial_state
        self.helper = helper
        self.decoder = decoder

    def _hidden_state(self, dec_cell):
        reuse = get_boolean(self.params["pass_hidden_state"].lower())
        if reuse:
            dec_init_state = dec_cell.zero_state(dtype=tf.float32, batch_size=self.dynamic_batch_size).clone(
                cell_state=self.enc_final_state)
        else:
            dec_init_state = dec_cell.zero_state(dtype=tf.float32, batch_size=self.dynamic_batch_size)

        return dec_init_state

    def _create_predictions(self, decoder_output):
        """Inference output with no beam search"""
        return tf.expand_dims(decoder_output.sample_id, -1, name='base_decoder_prediction')


    def _compute(self,dec_outputs):
        logits = tf.identity(dec_outputs.rnn_output, name='logits')  # logits: [batch_size x max_dec_len x dec_vocab_size+1]
        predictions = self._create_predictions(dec_outputs)
        return logits, predictions

    def _create_decoder_cell(self):
        return tf.contrib.rnn.MultiRNNCell(
            [self._make_cell(self.params["rnn_size"], self.params["keep_probability"]) for _ in
             range(self.params["num_layers"])])

    #TODO: look up the right scale for the initializer
    def _build_decoder(self, dec_cell, dec_embed_input, dec_param_list):
        self._setup_parameters(dec_param_list)
        scope = tf.get_variable_scope()
        scale = np.float(self.params["init_scale"])
        if self.initializer:
            scope.set_initializer(self.initializer(-scale, scale))
        else:
            scope.set_initializer(tf.random_uniform_initializer(-scale,scale))

        dec_init_state = self._hidden_state(dec_cell)
        output_layer = layers_core.Dense(self.vocab_size,
                                         use_bias=False,
                                         kernel_initializer=scope.initializer,
                                         name='output_projection')

        decoder_dict = self._create_decoder_object(dec_cell, dec_embed_input, dec_init_state, output_layer)

        self.init_decoder = get_boolean(self.params["init_decoder"])
        if not self.init_decoder:
            self._setup_decoder(dec_init_state, decoder_dict["helper"], decoder_dict["decoder"])

        # Create the weights for sequence_loss
        masks = tf.sequence_mask(self.summ_length, self.max_length, dtype=tf.float32, name='masks')

        # output and state at each time-step
        # self.train_dec_outputs shape is (64, ?, 9138)
        #TODO: look into why impute_finished needs to be false or else the model fails (different rank)
        dec_outputs, dec_last_state, _ = tf.contrib.seq2seq.dynamic_decode(
            self.decoder,
            output_time_major=False,
            impute_finished=False,
            swap_memory=True,
            maximum_iterations= self.max_length)

        logits, predictions = self._compute(dec_outputs)

        return (dec_outputs, dec_last_state, masks, logits, predictions)





