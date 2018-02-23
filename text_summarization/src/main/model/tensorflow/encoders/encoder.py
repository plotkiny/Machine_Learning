

"""
Some of the code is shamelessly plugged from Google's Seq2Seq implementation found below.
https://github.com/google/seq2seq
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple

import six

from main.model.tensorflow.configurable import Configurable

EncoderOutput = namedtuple(
    "EncoderOutput",
    "outputs final_state attention_values attention_values_length")


@six.add_metaclass(abc.ABCMeta)
class Encoder(Configurable):
  """Abstract encoder class. All encoders should inherit from this.

  Args:
    params: A dictionary of hyperparameters for the encoder.
    name: A variable scope for the encoder graph.
  """

  def __init__(self, params, mode, output_dir, name="base_encoder"):
    Configurable.__init__(self, params, mode, output_dir)

  def _build(self, inputs, *args, **kwargs):
    return self.encode(inputs, *args, **kwargs)

  @abc.abstractmethod
  def encode(self, *args, **kwargs):
    """
    Encodes an input sequence.

    Args:
      inputs: The inputs to encode. A float32 tensor of shape [B, T, ...].
      sequence_length: The length of each input. An int32 tensor of shape [T].

    Returns:
      An `EncoderOutput` tuple containing the outputs and final state.
    """
    raise NotImplementedError
