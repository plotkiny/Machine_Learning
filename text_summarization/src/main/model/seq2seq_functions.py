
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell_impl import _state_size_with_prefix


def _get_initial_cell_state(cell, initializer, batch_size, dtype):
    """Return state tensor(s), initialized with initializer.
    Args:
      cell: RNNCell.
      batch_size: int, float, or unit Tensor representing the batch size.
      initializer: function with two arguments, shape and dtype, that
          determines how the state is initialized.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` initialized
      according to the initializer.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        init_state_flat = [
            initializer(_state_size_with_prefix(s), batch_size, dtype, i)
            for i, s in enumerate(state_size_flat)]
        init_state = nest.pack_sequence_as(structure=state_size,
                                           flat_sequence=init_state_flat)
    else:
        init_state_size = _state_size_with_prefix(state_size)
        init_state = initializer(init_state_size, batch_size, dtype, None)

    return init_state

def _zero_state_initializer(shape, batch_size, dtype, index):
    z = tf.zeros(tf.stack(_state_size_with_prefix(shape, [batch_size])), dtype)
    z.set_shape(_state_size_with_prefix(shape, prefix=[None]))
    return z


def _make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index):
        args = kwargs.copy()

        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype

        var = tf.get_variable(**args)
        var = tf.expand_dims(var, 0)
        var = tf.tile(var, tf.stack([batch_size] + [1] * len(shape)))
        var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
        return var

    return variable_state_initializer


def _make_gaussian_state_initializer(initializer, deterministic_tensor=None, stddev=0.3):
    def gaussian_state_initializer(shape, batch_size, dtype, index):
        init_state = initializer(shape, batch_size, dtype, index)
        if deterministic_tensor is not None:
            return tf.cond(deterministic_tensor,
                           lambda: init_state,
                           lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev))
        else:
            return init_state + tf.random_normal(tf.shape(init_state), stddev=stddev)
    return gaussian_state_initializer


    #######