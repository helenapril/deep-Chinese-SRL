import tensorflow as tf
from highway_lstm import HighwayLSTMCell, MultiplicativeLSTMCell, MulIntLSTMCell
from variational_dropout_wrapper import VariationalDropoutWrapper

try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def linear(input_, output_size, scope=None):
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype,
                                    initializer=tf.constant_initializer(0.1))

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


class SRLModel():

    def __init__(self, inputs, preds, contexts, marks, labels, ori_inputs,
                 seq_length, vocab_size, embedding_size,
                 mark_embedding_size, num_classes, batch_size,
                 cell_name, hidden_size, num_layers, dropout_keep_proba,
                 dropout_mode, is_training):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dropout_keep_proba = dropout_keep_proba
        self.cell_name = cell_name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_mode = dropout_mode
        self.mark_embedding_size = mark_embedding_size
        self.is_training = is_training

        self.num_unroll_steps = tf.shape(inputs)[1]
        self.inputs = tf.reshape(inputs, shape=[self.batch_size, self.num_unroll_steps])
        self.ori_inputs = tf.reshape(ori_inputs, shape=[self.batch_size, self.num_unroll_steps])
        preds = tf.reshape(preds, shape=[self.batch_size, 1])
        contexts = tf.reshape(contexts, shape=[self.batch_size, 5])
        marks = tf.reshape(marks, shape=[self.batch_size, self.num_unroll_steps])
        self.labels = tf.reshape(labels, shape=[self.batch_size, self.num_unroll_steps])
        self.seq_length = tf.reshape(seq_length, shape=[self.batch_size])

        preds = tf.expand_dims(preds, axis=1)
        preds = tf.tile(preds, multiples=[1, self.num_unroll_steps, 1])
        contexts = tf.expand_dims(contexts, axis=1)
        contexts = tf.tile(contexts, multiples=[1, self.num_unroll_steps, 1])
        #inputs = tf.concat([inputs, preds, contexts], axis=2)

        with tf.variable_scope("embedding"):
            self.embedding_matrix = tf.get_variable(name="embedding_matrix",
                                                    shape=[self.vocab_size, self.embedding_size],
                                                    initializer=tf.random_normal_initializer(-0.05, 0.05),
                                                    dtype=tf.float32)
            self.embedding_placeholder = tf.placeholder(tf.float32,
                                                        [self.vocab_size, self.embedding_size])
            self.embedding_init = self.embedding_matrix.assign(self.embedding_placeholder)
            self.inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)
            inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)
            inputs_embedded = tf.reshape(inputs_embedded,
                                         shape=[self.batch_size, self.num_unroll_steps, self.embedding_size])
            self.embedding_mark_matrix = tf.get_variable(
                name="embedding_mark_matrix", shape=[3, self.mark_embedding_size],
                initializer=tf.random_normal_initializer(-0.05, 0.05), dtype=tf.float32)
            marks = tf.reshape(marks, shape=[self.batch_size, self.num_unroll_steps])
            mark_embedded = tf.nn.embedding_lookup(self.embedding_mark_matrix, marks)
            self.inputs_embedded = tf.concat([inputs_embedded, mark_embedded], axis=2)

        with tf.variable_scope('get_cell', initializer=tf.orthogonal_initializer):

            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True, forget_bias=1.0)

            def gru_cell():
                return tf.contrib.rnn.GRUCell(self.hidden_size)

            def dropout():
                if self.cell_name == 'lstm':
                    cell = lstm_cell()
                elif self.cell_name == 'highway':
                    cell = HighwayLSTMCell(self.hidden_size, state_is_tuple=True, forget_bias=1.0)
                elif self.cell_name == 'Multi':
                    cell = MultiplicativeLSTMCell(self.hidden_size, state_is_tuple=True, forget_bias=1.0)
                elif self.cell_name == 'MulInt':
                    cell = MulIntLSTMCell(self.hidden_size, forget_bias=1.0)
                else:
                    cell = gru_cell()
                if not dropout_mode:
                    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_proba)
                else:
                    return VariationalDropoutWrapper(cell, batch_size=self.batch_size, keep_prob=self.dropout_keep_proba)

        with tf.variable_scope('%s' % self.cell_name):
            '''with tf.variable_scope('single_layer'):
                cell = dropout()
                outputs, final_rnn_state = tf.nn.dynamic_rnn(
                    cell, self.inputs_embedded, sequence_length=self.seq_length, dtype=tf.float32)
                reverse_outputs = tf.reverse_sequence(outputs, seq_lengths=self.seq_length, seq_axis=1)
                next_layer_input = reverse_outputs'''

            next_layer_input = self.inputs_embedded
            for layer in range(self.num_layers):
                with tf.variable_scope('%d_layer' % layer):
                    with tf.variable_scope('forward', initializer=tf.orthogonal_initializer()):
                        cells_fw = dropout()
                    with tf.variable_scope('backward', initializer=tf.orthogonal_initializer()):
                        cells_bw = dropout()
                    outputs, final_rnn_state = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cells_fw,
                        cell_bw=cells_bw,
                        inputs=next_layer_input,
                        sequence_length=self.seq_length,
                        dtype=tf.float32)
                    outputs = tf.concat(outputs, 2)
                    outputs = tf.reshape(outputs,
                                         [self.batch_size, self.num_unroll_steps, 2*self.hidden_size])
                    #reverse_outputs = tf.reverse_sequence(outputs, seq_lengths=self.seq_length, seq_axis=1)
                    #next_layer_input = reverse_outputs
                    next_layer_input = outputs

            self.layer_output = outputs

        with tf.variable_scope("crf_loss"):
            layer_output = tf.reshape(self.layer_output, [-1, 2*self.hidden_size])
            scores = linear(layer_output, self.num_classes)
            self.scores = tf.reshape(scores, [self.batch_size, self.num_unroll_steps, self.num_classes])
            self.labels = tf.cast(self.labels, dtype=tf.int32)
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.scores, self.labels, self.seq_length)
            self.loss = tf.reduce_mean(-self.log_likelihood)

        '''with tf.variable_scope("crf_prediction"):
            self.layer_output = tf.reshape(self.layer_output, [-1, 2*self.hidden_size])
            self.scores = linear(self.layer_output, self.num_classes)
            self.scores = tf.reshape(self.scores, [self.batch_size, self.num_unroll_steps, self.num_classes])
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(
                self.scores, self.transition_params, self.seq_length)'''

        '''with tf.variable_scope("softmax_loss"):
            layer_output_flat = tf.reshape(self.layer_output, [-1, 2*self.hidden_size])
            scores_flat = linear(layer_output_flat, self.num_classes)
            self.scores = tf.reshape(scores_flat, [self.batch_size, self.num_unroll_steps, self.num_classes])
            inputs_flat = tf.reshape(self.inputs, [self.batch_size * self.num_unroll_steps])
            labels_flat = tf.reshape(self.labels, [self.batch_size * self.num_unroll_steps])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_flat, labels=labels_flat)
            mask = tf.sign(tf.to_float(inputs_flat))
            masked_loss = mask * losses
            self.loss = tf.reduce_sum(masked_loss)/tf.reduce_sum(mask)'''

