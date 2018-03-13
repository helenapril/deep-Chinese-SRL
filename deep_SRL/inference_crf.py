import numpy as np
import os
import cPickle


def get_transition_params(transition_params):
    with open(os.path.join('data', 'id_to_label.pkl'), 'rb') as f:
        vocab = cPickle.load(f)
    num_tags = len(vocab)
    label_strs = []
    for id in range(num_tags):
        label_strs.append(vocab[id])
    #transition_params = np.zeros([num_tags, num_tags], dtype=np.float32)
    for i, prev_label in enumerate(label_strs):
        for j, label in enumerate(label_strs):
            if i != j and label[0] == 'I' and not prev_label == 'B' + label[1:]:
                transition_params[i, j] = np.NINF
            if i == j and label[0] == 'B':
                transition_params[i, j] = np.NINF
            if prev_label[0] == 'I' and label[0] == 'B' + prev_label[1:]:
                transition_params[i, j] = np.NINF
    return transition_params


def viterbi_decode(score, transition_params):
    """ Adapted from Tensorflow implementation.
    Decode the highest scoring sequence of tags outside of TensorFlow.
    This should only be used at test time.
    Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
    indicies.
    viterbi_score: A float containing the score for the Viterbi sequence.
    """
    transition_params = get_transition_params(transition_params)
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]
    with open(os.path.join('data', 'id_to_label.pkl'), 'rb') as f:
        vocab = cPickle.load(f)
    num_tags = len(vocab)
    for id in range(num_tags):
        str_label = vocab[id]
        if str_label[0] == 'I':
            trellis[0][id] = np.NINF
    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)
    # bp[t][j] at time t, final state j , the t-1 state
    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


def recover(inputs, scores, golds, lens, matrix):
    new_inputs = []
    new_preds = []
    new_golds = []
    for line, score, gold, length in zip(inputs, scores, golds, lens):
        line = line[:length]
        pred = viterbi_decode(score, matrix)[0][:length]
        gold = gold[:length]
        #print line, pred, gold
        new_inputs.append(line)
        new_preds.append(pred)
        new_golds.append(gold)
    return new_inputs, new_preds, new_golds


if __name__ == "__main__":
    scores = np.ones(shape=[36, 36])
    #viterbi_decode(scores)
