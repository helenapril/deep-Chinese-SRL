import tensorflow as tf
import numpy as np
import os
import codecs
import time
from record_reader import TextLoader
from model_tmp import SRLModel
from eval import eval
from inference import recover
from collections import OrderedDict

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("mark_embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float('max_grad_norm',       5.0,  'normalize gradients at')
tf.flags.DEFINE_float('learning_rate',       0.001,  'starting learning rate')
tf.flags.DEFINE_float('learning_rate_decay', 0.5,  'learning rate decay')
tf.flags.DEFINE_integer("decay_steps", 3000, "Number of checkpoints to store (default: 5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1115, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_gpus", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_layers", 8, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_checkpoints", 180, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string('train_dir', 'check_attention', 'checkpoint')
tf.flags.DEFINE_string('mode', 'rnn', 'algorithm')
tf.flags.DEFINE_string('cell', 'lstm', 'algorithm')
tf.flags.DEFINE_boolean('dropout_mode', False, 'whether to share across time-step')


FLAGS = tf.flags.FLAGS


def main(_):

    valid_loader = TextLoader('valid', FLAGS.batch_size, 1, 1, None)
    batch_queue_data = valid_loader.batch_data
    writer = codecs.open('output_4', 'w', encoding='utf-8')
    print("Loading data...")
    # Training
    # ==================================================

    '''
        evaluate on dev dataset 
        compared to valid.py, add a choice to decide whether applying 
        layer_norm to network
    '''
    with tf.variable_scope('SRLModel'):
        valid_cnn = SRLModel(
            batch_queue_data[0],
            batch_queue_data[1],
            batch_queue_data[2],
            batch_queue_data[3],
            labels=batch_queue_data[4],
            ori_inputs=batch_queue_data[-1],
            seq_length=batch_queue_data[5],
            num_classes=36,
            vocab_size=381057,
            embedding_size=FLAGS.embedding_dim,
            mark_embedding_size=FLAGS.mark_embedding_dim,
            batch_size=FLAGS.batch_size,
            cell_name=FLAGS.cell,
            hidden_size=FLAGS.hidden_size,
            num_layers=FLAGS.num_layers,
            dropout_keep_proba=1.0,
            LN_mode=True,
            dropout_mode=False,
            is_training=False
        )

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)
        sess = tf.Session(config=session_conf)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt:
            files = os.listdir(FLAGS.train_dir)
            files = [os.path.join(FLAGS.train_dir, f) for f in files]
            check_point_files = []
            for f in files:
                if '.index' in f:
                    f = f[0:len(f) - 6]
                    check_point_files.append(f)
            num_batches_valid = int(1115 / FLAGS.batch_size)
            best_loss = None
            best_model = None
            acc_dict = {}
            for check_point in check_point_files:
                saver.restore(sess, check_point)
                global_step = check_point.split('/')[-1].split('-')[-1]
                global_step = int(global_step)
                print ("load model from ", check_point)
                loss = 0.0
                above = 0.0
                below = 0.0
                below_recall = 0.0
                for step in range(num_batches_valid):
                    inputs, scores, golds, lens, batch_loss = sess.run([
                        valid_cnn.ori_inputs,
                        valid_cnn.scores,
                        valid_cnn.labels,
                        valid_cnn.seq_length,
                        valid_cnn.loss])

                    loss += batch_loss
                    avg_cur_loss = loss / (step + 1)
                    inputs, preds, golds = recover(inputs, scores, golds, lens)
                    ab, ac, ad = eval(inputs, preds, golds, is_training=False)
                    above += ab
                    below += ac
                    below_recall += ad
                    if (step + 1) % 5 == 0:
                        if below and below_recall:
                            accuracy = float(above) / below
                            recall = float(above) / below_recall
                            if accuracy and recall:
                                F1 = 2.0 / (1.0 / accuracy + 1.0 / recall)
                                print('[%d/%d], avg_train_loss =%.4f, above = %.4f, below = %.4f, below_recall = %.4f, '
                                      'acc = %.4f, rec = %.4f, F1 = %.4f '
                                      % (step + 1, num_batches_valid, avg_cur_loss, above, below, below_recall, accuracy, recall, F1))
                            else:
                                print('[%d/%d], avg_train_loss =%.4f, above = %.4f, below = %.4f, below_recall = %.4f, '
                                      'acc = %.4f, rec = %.4f'
                                      % (step + 1, num_batches_valid, avg_cur_loss, above, below, below_recall, accuracy, recall))
                        else:
                            print('[%d/%d], avg_train_loss =%.4f, above = %.4f, below = %.4f, below_recall = %.4f, '
                                  % (step + 1, num_batches_valid, avg_cur_loss, above, below, below_recall))

                accuracy = float(above) / below
                recall = float(above) / below_recall
                F1 = 2.0 / (1.0 / accuracy + 1.0 / recall)
                print('global_step =%d, acc = %.4f, rec = %.4f, F1 = %.4f, '
                      % (global_step, accuracy, recall, F1))
                acc_dict[global_step] = F1

                if best_loss is None or best_loss < F1:
                    best_loss = F1
                    best_model = global_step

            sort_acc_dict = OrderedDict(sorted(acc_dict.items(), key=lambda d: d[0]))
            for k, v in sort_acc_dict.items():
                writer.write(str(k) + ' : ' + str(v) + '\n')
            print ('best_model from step %d, epoch:%d, F1: %f' % (best_model, best_model, best_loss))


if __name__ == "__main__":
    tf.app.run()
