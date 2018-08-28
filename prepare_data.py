import struct
import re
import numpy as np
from numpy import random
from random import shuffle
import pickle
from constants import *
import gensim
from keras.preprocessing.sequence import pad_sequences

def read_word_and_its_vec(opened_file, vec_len):
    try:
        char = opened_file.read(1)
        word = b''
        while char != b' ':
            word += char
            char = opened_file.read(1)
        vec = np.empty(vec_len)
        for i in range(vec_len):
            num = struct.unpack('f', opened_file.read(4))
            vec[i] = num[0]
        char = opened_file.read(1)
        word = word.decode()
    finally:
        return word, vec


def get_dict(dict_file):
    my_dict = open(dict_file, 'rb')
    line = my_dict.readline()
    line = line.split()
    row = int(line[0])
    col = int(line[1])
    result_dict = {}
    for _ in range(row):
        word, vec = read_word_and_its_vec(my_dict, col)
        result_dict[word] = vec
    my_dict.close()
    return result_dict, col


def word2vec(dictionary, word, vec_size):
    try:
        result = dictionary[word]
        return result
    except KeyError:
        new_vec = random.uniform(-1, 1, size = vec_size)
        result = dictionary[word] = new_vec
        return result


def corpora2vec(dictionary, corpora, vec_size):
    result = []
    for sent in corpora:
        curr = []
        for word in sent:
            curr.append(word2vec(dictionary, word, vec_size))
            # to test without softlink_ru (you also need to
            # comment the last line of this file)
            # curr.append(random.uniform(-1, 1, size = vec_size))
        result.append(curr)
    return result


def padd_sent(sent, vec_size, sent_size):
    res_sent = np.zeros([sent_size, vec_size])
    if len(sent) <= sent_size:
        res_sent[sent_size - len(sent):] = sent
    else: res_sent = sent[:sent_size]
    return res_sent


def padd_corpora(corpora, vec_size, sent_size):
    res_corpora = []
    for sent in corpora:
        res_corpora.append(padd_sent(sent,         \
                                     vec_size,     \
                                     sent_size))
    return res_corpora


def prepare_corpora(dictionary, corpora, vec_size, \
                    sent_size):
    '''
    takes a batch and prepare it
    corpora: list of words

    '''
    # corpora = del_empty(corpora)
    vec_dictionary = corpora2vec(dictionary, corpora, vec_size)
    vec_dictionary = padd_corpora(vec_dictionary, \
                                  vec_size,       \
                                  sent_size)
    return vec_dictionary    


def rand(vec_size):
    return normalize(normal(size = vec_size))


def get_data_seq2seq(corpora_file):
    input_texts = []
    target_texts = []
    corpora = open(corpora_file, 'r', encoding='utf-8')
    lines = corpora.read().split('\n')
    i = 0
    for line in lines:
        try:
            input_text, target_text = line.split('\t')
        except ValueError:
            pass
        target_text = 'SSTTAARRTT ' + target_text[:-1] + ' EENNDD'
        input_texts.append(input_text)
        target_texts.append(target_text)
    return input_texts, target_texts


def prepare_input_string(input_string):
    enc_texts = input_string.split()
    enc_texts_vec = prepare_corpora(enc_dict,
                                    enc_texts,
                                    enc_vec_size,
                                    enc_sent_size)
    return enc_texts_vec

    
def next_batch_keras(input_texts, target_texts, n,
                     enc_vec_size, dec_vec_size,
                     enc_sent_size, dec_sent_size,
                     tokenizer):
    '''
    for keras eng-rus files (seq2seq lstm word2vec)
    '''
    while True:
        i = 0
        while i < len(input_texts):
            enc_texts = [sent.split() for sent in input_texts[i:i+n]]
            dec_texts = [sent.split() for sent in target_texts[i:i+n]]
            enc_texts_vec = prepare_corpora(enc_dict,
                                            enc_texts,
                                            enc_vec_size,
                                            enc_sent_size)
            dec_texts_vec = prepare_corpora(dec_dict,
                                            dec_texts,
                                            dec_vec_size,
                                            dec_sent_size)
            target = [i[1:] for i in dec_texts]
            target = tokenizer.texts_to_sequences(target)
            target = pad_sequences(target, maxlen=dec_sent_size)
            yield [np.array(enc_texts_vec),
                   np.array(dec_texts_vec)], target
            i += n


# # building a dictionary
# model = gensim.models.KeyedVectors.load_word2vec_format(
#     './softlink_en_big', binary=True)
# dictionary = {}
# for key in model.vocab:
#     if str.isalpha(key):
#         dictionary[key.lower()] = model.wv[key]

# for testing without dictionaries
enc_dict = dec_dict = []


# comment this to test the model without dictionary
enc_dict, enc_vec_size = get_dict(en_dict_source)
dec_dict, dec_vec_size = get_dict(ru_dict_source)
