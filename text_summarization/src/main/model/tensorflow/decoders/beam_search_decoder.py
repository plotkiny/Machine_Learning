
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from main.model.tensorflow.decoders.basic_decoder import BasicDecoder

class BeamSearchDecoder(BasicDecoder):

    def __init__(self, decoder, config):
        super(BeamSearchDecoder, self).__init__(decoder.params, decoder.mode, decoder.output_dir,
                                                config.vocab_size, config.summ_length, config.max_length,
                                                name="beam_search_decoder")
        self.decoder = decoder
        self.config = config

    def _create_decoder_object(self, dec_cell, _dec_embed_input, dec_init_state, output_layer):
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=dec_cell,
                    initial_state=dec_init_state,
                    output_layer=output_layer,
                    embedding=self.config.embed_matrix,
                    start_tokens=self.config.start_token,
                    end_token=self.config.end_token,
                    beam_width=self.config.beam_width,
                    length_penalty_weight=self.config.length_penalty_weight)
        return {"helper":None, "decoder":decoder}

    def _create_predictions(self, decoder_output):
        """Inference output with beam search"""
        return tf.identity(decoder_output.predicted_ids, name="beam_search_decoder_prediction")

    def _compute(self, decoder_output):
        logits = tf.no_op()
        predictions = self._create_predictions(decoder_output)
        return logits, predictions



