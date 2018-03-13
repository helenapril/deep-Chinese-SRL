import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell import RNNCell
try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


class HighwayLSTMCell(RNNCell):
    '''
    add highway connection to lstm cell
    '''

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None):

        super(HighwayLSTMCell, self).__init__(_reuse=reuse)
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

        new_c = (
            c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)
        high_h = sigmoid(t) * new_h + (1.0 - sigmoid(t)) * self._activation(trans_input)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, high_h)
        else:
            new_state = tf.concat([new_c, high_h], 1)
        return high_h, new_state


class MultiplicativeLSTMCell(RNNCell):
    """Multiplicative LSTM.
       Ben Krause, Liang Lu, Iain Murray, and Steve Renals,
       "Multiplicative LSTM for sequence modelling, "
       in Workshop Track of ICLA 2017,
       https://openreview.net/forum?id=SJCS5rXFl&noteId=SJCS5rXFl
    """

    def __init__(self, num_units,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=tf.orthogonal_initializer(),
                 num_proj=None,
                 proj_clip=None,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tf.tanh,
                 reuse=None):
        super(MultiplicativeLSTMCell, self).__init__(_reuse=reuse)
        '''Initialize the parameters for an LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          use_peepholes: bool, set True to enable diagonal/peephole
            connections.
          cell_clip: (optional) A float value, if provided the cell state
            is clipped by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight
            matrices.
          num_proj: (optional) int, The output dimensionality for
            the projection matrices.  If None, no projection is performed.
          forget_bias: Biases of the forget gate are initialized by default
            to 1 in order to reduce the scale of forgetting at the beginning of
            the training.
          activation: Activation function of the inner states.
        '''
        self.num_units = num_units
        self.use_peepholes = use_peepholes
        self.cell_clip = cell_clip
        self.num_proj = num_proj
        self.proj_clip = proj_clip
        self.initializer = initializer
        self.forget_bias = forget_bias
        self.state_is_tuple = state_is_tuple
        self.activation = activation

        if num_proj:
            self._state_size = (
                tf.nn.rnn_cell.LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                tf.nn.rnn_cell.LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):

        num_proj = self.num_units if self.num_proj is None else self.num_proj

        if self.state_is_tuple:
            (c_prev, h_prev) = state
        else:
            c_prev = tf.slice(state, [0, 0], [-1, self.num_units])
            h_prev = tf.slice(state, [0, self.num_units], [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]

        with tf.variable_scope(scope or type(self).__name__, initializer=self.initializer):
            if input_size.value is None:
                raise ValueError(
                    "Could not infer input size from inputs.get_shape()[-1]")

            with tf.variable_scope("Multipli_Weight"):
                concat = _linear([inputs, h_prev], 2 * self.num_units, True)
            Wx, Wh = tf.split(concat, 2, 1)
            m = Wx * Wh  # equation (18)

            with tf.variable_scope("LSTM_Weight"):
                lstm_matrix = _linear([inputs, m], 4 * self.num_units, True)
            i, j, f, o = tf.split(lstm_matrix, 4, 1)

            # Diagonal connections
            if self.use_peepholes:
                w_f_diag = tf.get_variable(
                    "W_F_diag", shape=[self.num_units], dtype=dtype)
                w_i_diag = tf.get_variable(
                    "W_I_diag", shape=[self.num_units], dtype=dtype)
                w_o_diag = tf.get_variable(
                    "W_O_diag", shape=[self.num_units], dtype=dtype)

            if self.use_peepholes:
                c = c_prev * tf.sigmoid(f + self.forget_bias + w_f_diag * c_prev) + \
                    tf.sigmoid(i + w_i_diag * c_prev) * j
            else:
                c = c_prev * tf.sigmoid(f + self.forget_bias) + \
                    tf.sigmoid(i) * j

            if self.cell_clip is not None:
                c = tf.clip_by_value(c, -self.cell_clip, self.cell_clip)

            if self.use_peepholes:
                h = tf.sigmoid(o + w_o_diag * c) * \
                    self.activation(c * (o + w_o_diag * c))
            else:
                h = self.activation(c * o)

            if self.num_proj is not None:
                w_proj = tf.get_variable(
                    "W_P", [self.num_units, num_proj], dtype=dtype)

                h = tf.matmul(h, w_proj)
                if self.proj_clip is not None:
                    h = tf.clip_by_value(h, -self.proj_clip, self.proj_clip)

            new_state = (tf.nn.rnn_cell.LSTMStateTuple(c, h)
                         if self.state_is_tuple else tf.concat([c, h],1))

            return h, new_state


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term


class MulIntLSTMCell(RNNCell):
    """Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/pdf/1409.2329v5.pdf.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    Biases of the forget gate are initialized by default to 1 in order to reduce
    the scale of forgetting in the beginning of the training.
    """

    def __init__(self, num_units, forget_bias=1.0,
                 use_highway=False, num_highway_layers=2,
                 use_recurrent_dropout=False, recurrent_dropout_factor=0.90):
        super(MulIntLSTMCell, self).__init__(_reuse=None)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self.use_highway = use_highway
        self.num_highway_layers = num_highway_layers
        self.use_recurrent_dropout = use_recurrent_dropout
        self.recurrent_dropout_factor = recurrent_dropout_factor

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = state
            concat = multiplicative_integration([inputs, h], self._num_units * 4, 0.0)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(concat, num_or_size_splits=4, axis=1)

            if self.use_recurrent_dropout:
                input_contribution = tf.nn.dropout(tf.tanh(j), self.recurrent_dropout_factor)
            else:
                input_contribution = tf.tanh(j)

            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * input_contribution
            new_h = tf.tanh(new_c) * tf.sigmoid(o)
            new_state = LSTMStateTuple(new_c, new_h)

        return new_h, new_state  # purposely reversed


def multiplicative_integration(list_of_inputs, output_size, initial_bias_value=0.0,
                               weights_already_calculated=False, use_highway_gate=False,
                               use_l2_loss=False, scope=None):
    '''expects len(2) for list of inputs and will perform integrative multiplication
    weights_already_calculated will treat the list of inputs as Wx and Uz and is useful for batch normed inputs
    '''
    with tf.variable_scope(scope or 'double_inputs_multiple_integration'):
        if len(list_of_inputs) != 2:
            raise ValueError('list of inputs must be 2, you have:', len(list_of_inputs))

        if weights_already_calculated:  # if you already have weights you want to insert from batch norm
            Wx = list_of_inputs[0]
            Uz = list_of_inputs[1]

        else:
            with tf.variable_scope('Calculate_Wx_mulint', initializer=tf.orthogonal_initializer()):
                Wx = _linear(list_of_inputs[0], output_size, True)
            with tf.variable_scope("Calculate_Uz_mulint", initializer=tf.orthogonal_initializer()):
                Uz = _linear(list_of_inputs[1], output_size, True)

        with tf.variable_scope("multiplicative_integration"):
            alpha = tf.get_variable('mulint_alpha', [output_size],
                                    initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.1))

            beta1, beta2 = tf.split(tf.get_variable('mulint_params_betas', [output_size * 2],
                                    initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.1)),
                                    num_or_size_splits=2, axis=0)

            original_bias = tf.get_variable('mulint_original_bias', [output_size],
                                            initializer=tf.truncated_normal_initializer(mean=initial_bias_value,
                                                                                        stddev=0.1))

        final_output = tf.tanh(alpha * Wx * Uz + beta1 * Uz + beta2 * Wx + original_bias)

    return final_output
