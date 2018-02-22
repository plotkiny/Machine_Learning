

"""
Some of the code is shamelessly plugged from Google's Seq2Seq implementation in the link below.
https://github.com/google/seq2seq
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc, six, yaml
import tensorflow as tf


class abstractstaticmethod(staticmethod):  #pylint: disable=C0111,C0103
  """Decorates a method as abstract and static"""
  __slots__ = ()

  def __init__(self, function):
    super(abstractstaticmethod, self).__init__(function)
    function.__isabstractmethod__ = True

  __isabstractmethod__ = True


@six.add_metaclass(abc.ABCMeta)
class Configurable(object):

  """
  Interface for all classes that are configurable via a parameters dictionary.

  Args:
    params:A configuration file.
    mode: A value denoting training or inference
  """

  def __init__(self, params, mode, output_dir):
    self._params = params
    self._mode = mode
    self.output_dir = output_dir
    self._print_params()

  def _print_params(self):
    """Logs parameter values"""
    classname = self.__class__.__name__
    tf.logging.info("Creating %s in mode=%s", classname, self._mode)
    tf.logging.info("\n%s", yaml.dump({classname: self._params}))

  @property
  def mode(self):
    """Returns a value in tf.contrib.learn.ModeKeys.
    """
    return self._mode

  @property
  def params(self):
    """Returns a dictionary of parsed parameters.
    """
    return self._params

  # @abstractstaticmethod
  # def default_params():
  #   """Returns a dictionary of default parameters. The default parameters
  #   are used to define the expected type of passed parameters. Missing
  #   parameter values are replaced with the defaults returned by this method.
  #   """
  #   raise NotImplementedError
