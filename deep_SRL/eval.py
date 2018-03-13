#!/usr/bin/python
#-*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cPickle


def convert(inputs, labels, is_training):
    '''

    :param inputs: word_id
    :param labels: label_id
    :param is_training:
    :return: convert id to original word
    '''
    if is_training:
        word_vocab_file = os.path.join('data', 'train_id_to_word.pkl')
        with open(word_vocab_file, 'rb') as f:
            word_vocab = cPickle.load(f)
    else:
        word_vocab_file = os.path.join('data', 'valid_id_to_word.pkl')
        with open(word_vocab_file, 'rb') as f:
            word_vocab = cPickle.load(f)

    label_vocab_file = os.path.join('data', 'id_to_label.pkl')
    with open(label_vocab_file, 'rb') as f:
        label_vocab = cPickle.load(f)
    new_inputs = []
    for line, label in zip(inputs, labels):
        new_input = []
        for word_id, label_id in zip(line, label):
            new_input.append(word_vocab[word_id] + '/' + label_vocab[label_id])
        new_inputs.append(new_input)
    return new_inputs


def eval(inputs, pred_labels, gold_labels, is_training):
    '''
    :param inputs: sentence
    :param pred_labels: labels predicted by model
    :param gold_labels: True labels
    :param is_training: convert id to words based on train_vocab or valid_vocab
    :return: recall, precision, F1
    '''
    case_true, case_recall, case_precision = 0, 0, 0
    golds = convert(inputs, gold_labels, is_training)
    preds = convert(inputs, pred_labels, is_training)
    assert len(golds) == len(preds), "length of prediction file and gold file should be the same."
    for gold, pred in zip(golds, preds):
        lastname = ''
        keys_gold, keys_pred = {}, {}
        for item in gold:
            word, label = item.split('/')[0], item.split('/')[-1]
            flag, name = label[:label.find('-')], label[label.find('-') + 1:]
            if flag == 'O':
                continue
            if flag == 'S':
                if name not in keys_gold:
                    keys_gold[name] = [word]
                else:
                    keys_gold[name].append(word)
            else:
                if flag == 'B':
                    if name not in keys_gold:
                        keys_gold[name] = [word]
                    else:
                        keys_gold[name].append(word)
                    lastname = name
                elif flag == 'I' or flag == 'E':
                    assert name == lastname, "the I-/E- labels are inconsistent with B- labels in gold file."
                    keys_gold[name][-1] += ' ' + word

        for item in pred:
            word, label = item.split('/')[0], item.split('/')[-1]
            flag, name = label[:label.find('-')], label[label.find('-') + 1:]
            if flag == 'O':
                continue
            if flag == 'S':
                if name not in keys_pred:
                    keys_pred[name] = [word]
                else:
                    keys_pred[name].append(word)
            else:
                if flag == 'B':
                    if name not in keys_pred:
                        keys_pred[name] = [word]
                    else:
                        keys_pred[name].append(word)
                    lastname = name
                elif flag == 'I' or flag == 'E':
                    assert name == lastname, "the I-/E- labels are inconsistent with B- labels in pred file."
                    keys_pred[name][-1] += ' ' + word

        for key in keys_gold:
            case_recall += len(keys_gold[key])
        for key in keys_pred:
            case_precision += len(keys_pred[key])

        for key in keys_pred:
            if key in keys_gold:
                for word in keys_pred[key]:
                    if word in keys_gold[key]:
                        case_true += 1
                        keys_gold[key].remove(word)  # avoid replicate words
    return case_true, case_precision, case_recall

