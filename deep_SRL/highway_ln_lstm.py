import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell import RNNCell
try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
from highway_lstm import _linear
from layer_norm import ln


class LNHighwayLSTMCell(RNNCell):
    '''
       add layer_norm to highway_lstm cell
    '''

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None):

        super(LNHighwayLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._weight_matrix = None
        self._trans_input = None

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):

        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        with tf.variable_scope("Weight", initializer=tf.orthogonal_initializer()):
            weight_matrix = _linear([inputs, h], 5 * self._num_units, True)
        with tf.variable_scope("transform_input", initializer=tf.orthogonal_initializer()):
            trans_input = _linear([inputs], self._num_units, True)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate batch_size * dim
        i, j, f, o, t = tf.split(weight_matrix, num_or_size_splits=5, axis=1)
        i = ln(i, scope='i_LN')
        j = ln(j, scope='j_LN')
        f = ln(f, scope='f_LN')
        o = ln(o, scope='o_LN')
        t = ln(t, scope='t_LN')
        new_c = (
            c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(ln(new_c, scope='new_c_LN')) * sigmoid(o)
        high_h = sigmoid(t) * new_h + \
                 (1.0 - sigmoid(t)) * self._activation(ln(trans_input, scope='new_input_LN'))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, high_h)
        else:
            new_state = tf.concat([new_c, high_h], 1)
        return high_h, new_state


class LNLSTMCell(RNNCell):
    '''
           add layer_norm to lstm cell
    '''

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None):

        super(LNLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._weight_matrix = None
        self._trans_input = None

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):

        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        with tf.variable_scope("Weight", initializer=tf.orthogonal_initializer()):
            weight_matrix = _linear([inputs, h], 4 * self._num_units, True)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate batch_size * dim
        i, j, f, o = tf.split(weight_matrix, num_or_size_splits=4, axis=1)
        i = ln(i, scope='i_LN')
        j = ln(j, scope='j_LN')
        f = ln(f, scope='f_LN')
        o = ln(o, scope='o_LN')
        new_c = (
            c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(ln(new_c, scope='new_c_LN')) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c,new_h], 1)
        return new_h, new_state


