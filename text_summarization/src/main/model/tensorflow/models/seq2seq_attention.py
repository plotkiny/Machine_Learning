
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from main.model.tensorflow.models.seq2seq_model import BasicSeq2Seq

__all__ = ["AttentionModel"]

class AttentionModel(BasicSeq2Seq):

    def __init__(self, params, mode, output_dir, name="att_seq2seq"):
        super(AttentionModel, self).__init__(params, mode, output_dir)

    def _create_attention_mechanism(self, attn_type, attn_size, memory, sequence_length):

        attn_options = ["luong", "scaled.luong", "bahdanau", "normed.bahdanau"]

        if attn_type == "luong":
            attn_mech = tf.contrib.seq2seq.LuongAttention(
                attn_size, memory, memory_sequence_length=sequence_length)
        elif attn_type == "scaled.luong":
            attn_mech = tf.contrib.seq2seq.LuongAttention(
                attn_size,
                memory,
                memory_sequence_length=sequence_length,
                scale=True)
        elif attn_type == "bahdanau":
            attn_mech = tf.contrib.seq2seq.BahdanauAttention(
                attn_size, memory, memory_sequence_length=sequence_length)
        elif attn_type == "normed.bahdanau":
            attn_mech = tf.contrib.seq2seq.BahdanauAttention(
                attn_size,
                memory,
                memory_sequence_length=sequence_length,
                normalize=True)
        else:
            raise ValueError(
                "Unknown attention option. The following attention types are currently supported %s" % attn_options)

        return attn_mech, attn_size

    def _create_decoder_cell(self, enc_output, seq_length):

        dec_cell = tf.contrib.rnn.MultiRNNCell([
            self._make_cell(self.params["rnn_size"]) for _ in range(self.params["num_layers"])])

        attn_type = self.params["attn_type"]
        attn_size = self.params["attn_size"]

        attn_mech, attn_size = self._create_attention_mechanism(attn_type, attn_size, enc_output, seq_length)

        dec_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=dec_cell,
            attention_mechanism=attn_mech,
            attention_layer_size=attn_size)

        return dec_cell





