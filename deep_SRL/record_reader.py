import codecs
import os
import cPickle
import tensorflow as tf


def make_example(sequence, preds, contexts, marks, labels, train_sequence):
    '''

    :param sequence: word_id feature
    :param preds: pred_id feature
    :param contexts: context feature
    :param marks: region_mark_id feature
    :param labels: label_id
    :return: feature_list
    '''
    input_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[token]))
        for token in sequence]
    pred_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[pred]))
        for pred in preds]
    context_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[context]))
        for context in contexts]
    mark_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[mark]))
        for mark in marks]
    label_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        for label in labels]
    length_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[len(sequence)]))
    ]
    train_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[token]))
        for token in train_sequence]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'preds': tf.train.FeatureList(feature=pred_features),
        'contexts': tf.train.FeatureList(feature=context_features),
        'marks': tf.train.FeatureList(feature=mark_features),
        'labels': tf.train.FeatureList(feature=label_features),
        'length': tf.train.FeatureList(feature=length_features),
        'train': tf.train.FeatureList(feature=train_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)


def parse(seq, pred_id, ctx_len):
    #[pred_id-ctx_len, pred_id+ctx_len]
    ctx = []
    mark = []
    seq_len = len(seq)
    for _ in range(max(0, ctx_len - pred_id)):
        ctx.append('ctx-pad')
    for id, word in enumerate(seq):
        if abs(id - pred_id) <= ctx_len:
            ctx.append(word)
            mark.append(2)
        else:
            mark.append(1)
    for _ in range(max(0, ctx_len + pred_id - seq_len)):
        ctx.append('ctx-pad')
    return ctx, mark


class TextLoader():
    '''
       read data using tensorflow input pipline,
       here we use sequence_example
    '''
    def __init__(self, filename, batch_size, num_threads, num_gpus, num_epochs):

        if filename == 'train':
            input_file = os.path.join('data', 'process_cpbtrain')
        if filename == 'valid':
            input_file = os.path.join('data', 'process_cpbdev')
        if filename == 'test':
            input_file = os.path.join('data', 'cpbtest.txt')
        self.vocab_file = os.path.join('data', "word_to_id.pkl")
        self.label_dict = os.path.join('data', "label_to_id.pkl")
        file_1 = os.path.join('data', 'word_train_tfrecord')
        file_2 = os.path.join('data', 'word_valid_tfrecord')
        file_3 = os.path.join('data', 'word_eval_tfrecord')
        if filename == 'train':
            files = [file_1]
        if filename == 'valid':
            files = [file_2]
        if filename == 'test':
            files = [file_3]
        #self.preprocess(input_file, file_2)
        self.batch_data = self.read_process(files, batch_size, num_threads, num_gpus, num_epochs)

    def preprocess(self, input_file, write_to_file):
        with open(self.vocab_file, 'rb') as f:
            vocab = cPickle.load(f)
        with open('data/valid_word_to_id.pkl', 'rb') as f:
            train_vocab = cPickle.load(f)
        with open(self.label_dict, 'rb') as f:
            label_dict = cPickle.load(f)
        self.vocab_size = len(vocab)
        print (self.vocab_size)
        print (len(label_dict))
        num = 0
        writer = tf.python_io.TFRecordWriter(write_to_file)
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                if not len(line):
                    continue
                num += 1
                content = []
                pred = []
                label = []
                for id, labeled_word in enumerate(line):
                    labeled_word = labeled_word.split('/')
                    content.append(labeled_word[0])
                    label.append(labeled_word[2])
                    if labeled_word[2] == 'rel':
                        pred_id = id
                        pred.append(labeled_word[0])
                context, mark = parse(content, pred_id, 2)

                sequence = [vocab[word] if word in vocab else vocab['unk'] for word in content]
                train_sequence = [train_vocab[word] for word in content]
                label_sequence = [label_dict[word] for word in label]
                context_sequence = [vocab[word] if word in vocab else vocab['unk'] for word in context]
                pred_sequence = [vocab[word] if word in vocab else vocab['unk'] for word in pred]
                ex = make_example(sequence, pred_sequence, context_sequence, mark, label_sequence, train_sequence)
                writer.write(ex.SerializeToString())
                if num % 1000 == 0:
                    print num
        writer.close()
        print (num, self.vocab_size)
        print("Wrote to {}".format(write_to_file))

    def read_process(self, filename, batch_size, num_threads, num_gpus, max_epochs):
        reader = tf.TFRecordReader()
        file_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=max_epochs)
        key, serialized_example = reader.read(file_queue)
        sequence_features = {
            "inputs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "preds": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "contexts": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "marks": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "length": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "train": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        # Parse the example (returns a dictionary of tensors)
        _, sequence_parsed = tf.parse_single_sequence_example(
            serialized_example,
            sequence_features=sequence_features
        )
        input_tensors = [sequence_parsed['inputs'], sequence_parsed['preds'],
                         sequence_parsed['contexts'], sequence_parsed['marks'],
                         sequence_parsed['labels'], sequence_parsed['length'],
                         sequence_parsed['train']]
        return tf.train.batch(
            input_tensors,
            batch_size=batch_size,
            capacity=10 + num_gpus * batch_size,
            num_threads=num_threads,
            dynamic_pad=True,
            allow_smaller_final_batch=False
        )


if __name__ == "__main__":
    loader = TextLoader('valid', 1, 1, 1, 7)
    batch_queue_data = loader.batch_data

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for _ in range(1):
        batch_data = sess.run(batch_queue_data)
        print (batch_data[-1])

