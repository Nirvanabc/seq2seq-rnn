import struct
import re
import numpy as np
from numpy.random import normal
from random import shuffle
import pickle
from constants import *
import gensim

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


def normalize(vec):
    total = sum(vec)
    return list(map(lambda x: x/total, vec))


def word2vec(word, vec_size):
    try:
        result = dictionary[word]
        return result
    except KeyError:
        new_vec = normalize(normal(size = vec_size))
        result = dictionary[word] = new_vec
        return result


def corpora2vec(corpora, vec_size):
    result = []
    for sent in corpora:
        curr = []
        for word in sent:
            # curr.append(word2vec(word, vec_size))
            # to test without softlink_ru (you also need to
            # comment the last line of this file)
            curr.append(normalize(normal(size = vec_size)))
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


def prepare_corpora(corpora, vec_size, \
                    sent_size):
    '''
    takes a batch and prepare it
    '''
    # corpora = del_empty(corpora)
    vec_dictionary = corpora2vec(corpora, vec_size)
    vec_dictionary = padd_corpora(vec_dictionary, \
                                  vec_size,       \
                                  sent_size)
    return vec_dictionary    


def rand(vec_size):
    return normalize(normal(size = vec_size))


def next_batch_keras(corpora_file, n, vec_size):
    '''
    for keras eng-rus files (seq2seq lstm word2vec)
    '''
    corpora = open(corpora_file, 'r', encoding='utf-8')
    while True:
        input_texts = []
        target_texts = []
        for _ in range(n):
            line = corpora.readline()
            if len(line) == 0:
                corpora = open(corpora_file, 'r',
                               encoding='utf-8')
                line = corpora.readline()
            input_text, target_text = line.split('\t')
            target_text = 'START_ ' + target_text[:-1] + ' _END'
            input_texts.append(input_text)
            target_texts.append(target_text)
        input_texts = prepare_corpora(input_text,
                                      vec_size,
                                      sent_size)
        target_texts = prepare_corpora(target_text,
                                       vec_size,
                                       sent_size)
            
        yield input_texts, target_texts
            

def next_batch(corpora_file, n, vec_size):
    '''
    input:
    corpora_file: file path
    n: size of returned batch
    
    return:
    batch of size n
    or StopIteration when reach the end
    of the file
    '''
    
    corpora = open(corpora_file, 'r')
    # until we reach the end of file
    while True:
        batch = []
        labels = []
        for _ in range(n):
            sent = corpora.readline()
            if len(sent) == 0:
                return
            sent = sent.split()
            if sent[:-1] != []:
                labels.append(int(sent[-1]))
                batch.append(sent[:-1])
        batch = prepare_corpora(batch, vec_size, sent_size)
        labels = [[1 - labels[i],
                   labels[i]] for i in range(len(labels))]
        batch = [batch, labels]
        yield batch
        

def generate_arrays_from_file(corpora_file, n = batch_size):
    '''
    for keras model.fit_generator
    '''
    
    corpora = open(corpora_file, 'r')
    
    # until we reach the end of file
    while True:
        batch = []
        labels = []
        for _ in range(n):
            sent = corpora.readline()
            if len(sent) == 0:
                corpora = open(corpora_file, 'r')
                sent = corpora.readline()
            sent = sent.split()
            if sent[:-1] != []:
                labels.append(int(sent[-1]))
                batch.append(sent[:-1])
        batch = np.array(prepare_corpora(batch,
                                         vec_size,
                                         sent_size))
        labels = np.array([[1 - labels[i],
                   labels[i]] for i in range(len(labels))])
        yield(batch, labels)
            
        
# # building a dictionary
# model = gensim.models.KeyedVectors.load_word2vec_format(
#     './softlink_en_big', binary=True)
# dictionary = {}
# for key in model.vocab:
#     if str.isalpha(key):
#         dictionary[key.lower()] = model.wv[key]

# dictionary, vec_size = get_dict(en_dict_source)
