import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell
try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def get_dropout_mask(keep_prob, shape):
    keep_prob = tf.convert_to_tensor(keep_prob, dtype=tf.float32)
    random_tensor = keep_prob + tf.random_uniform(shape, dtype=tf.float32)
    binary_tensor = tf.floor(random_tensor)
    dropout_mask = tf.div(1.0, keep_prob) * binary_tensor
    return dropout_mask


class VariationalDropoutWrapper(RNNCell):
    '''
    dropout sharing between time steps
    '''
    def __init__(self, cell, batch_size, keep_prob):
        super(VariationalDropoutWrapper, self).__init__(_reuse=None)
        self._cell = cell
        self._output_dropout_mask = get_dropout_mask(keep_prob, [batch_size, cell.output_size])
        self._state_dropout_mask = get_dropout_mask(keep_prob, [batch_size,  cell.output_size])

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        c, h = state
        h *= self._state_dropout_mask
        state = LSTMStateTuple(c, h)
        output, new_state = self._cell(inputs, state, scope)
        output *= self._output_dropout_mask
        return output, new_state
