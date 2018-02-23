

"""
Some of the code is shamelessly plugged from Google's Seq2Seq implementation found below.
https://github.com/google/seq2seq
"""

import copy
import tensorflow as tf
from main.model.tensorflow.encoders.encoder import Encoder, EncoderOutput
from main.model.tensorflow.training import utils as training_utils


def _default_rnn_cell_params():
  """Creates default parameters used by multiple RNN encoders.
  """
  return {
      "cell_class": "BasicLSTMCell",
      "cell_params": {
          "num_units": 128
      },
      "dropout_input_keep_prob": 1.0,
      "dropout_output_keep_prob": 1.0,
      "num_layers": 1,
      "residual_connections": False,
      "residual_combiner": "add",
      "residual_dense": False
  }


def _unpack_cell(cell):
  """Unpack the cells because the stack_bidirectional_dynamic_rnn
  expects a list of cells, one per layer."""
  if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
    return cell._cells  #pylint: disable=W0212
  else:
    return [cell]

def toggle_dropout(params, mode):
  """Disables dropout during eval/inference mode
  """
  cell_params = copy.deepcopy(params)
  if mode != "train":
    cell_params["dropout_input_keep_prob"] = 1.0
    cell_params["dropout_output_keep_prob"] = 1.0
  return cell_params



class BidirectionalRNNEncoder(Encoder):

    def __init__(self, params, mode, output_dir, name="bidirectional_rnn_encoder"):
        super(BidirectionalRNNEncoder, self).__init__(params, mode, output_dir)
        self.params["rnn_cell"] = toggle_dropout(self.params, self.mode)

    @staticmethod
    def default_params():
        return {
            "rnn_cell": _default_rnn_cell_params(),
            "init_scale": 0.04,
        }

    def encode(self, inputs, sequence_length):

        cell_fw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        cell_bw = training_utils.get_rnn_cell(**self.params["rnn_cell"])

        #TODO: figure out why stack_bidirectional_dynamic_rnn doesn't work
        #enc_state is a tuple of forward and backward (output_state_fw, output_state_bw)
        enc_output, enc_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            dtype=tf.float32,
            sequence_length=sequence_length)

        #concatenate the forward and backward outputs
        outputs_concat = tf.concat(enc_output, 2)

        return EncoderOutput(
            outputs=outputs_concat,
            final_state=enc_states,
            attention_values=outputs_concat,
            attention_values_length=sequence_length)
