

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from main.model.tensorflow.seq2seq.rnn_cell import MultiRNNCell, ExtendedMultiRNNCell
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn

def _get_cell_class(cell_class):
    if cell_class == "RNNCell":
        pass
    elif cell_class == "LSTMCell":
        return LSTMCell

def get_rnn_cell(residual_connections=False,
                 residual_combiner="add",
                 residual_dense=False,
                 **params):

  encoder_type = _get_cell_class(params["cell.class"])
  dropout_input_keep_prob = params["dropout.input.keep.prob"]
  dropout_output_keep_prob = params["dropout.output.keep.prob"]

  init_scale = np.float(params["init.scale"])
  scope = tf.get_variable_scope()
  scope.set_initializer(tf.random_uniform_initializer(-init_scale, init_scale))

  cells = []
  for _ in range(params["num.layers"]):
    cell = encoder_type(params["rnn.size"], initializer=scope.initializer)
    if dropout_input_keep_prob < 1.0 or dropout_output_keep_prob < 1.0:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell=cell,
          input_keep_prob=dropout_input_keep_prob,
          output_keep_prob=dropout_output_keep_prob)
    cells.append(cell)

  if len(cells) > 1:
    final_cell = ExtendedMultiRNNCell(
        cells=cells,
        residual_connections=residual_connections,
        residual_combiner=residual_combiner,
        residual_dense=residual_dense)
  else:
    final_cell = MultiRNNCell(cells=cells)

  return final_cell