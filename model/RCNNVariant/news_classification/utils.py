'''
AUTHOR :li peng cheng

DATE :2021/09/06 9:05
'''
import numpy as np
import pandas as pd
from torchtext.vocab import Vocab
from collections import Counter, OrderedDict
from gensim.models import Word2Vec

def get_vocab(corpus):
    print('获得词典中......')
    sentence = ''
    for i in range(len(corpus)):
        sentence = sentence + corpus.iloc[i][1] + ' '
        if i % 10000==0:
            print('%d句子已经处理' % (i))
    return Vocab(Counter(sentence.strip().split()))

def obtain_data(vocab, data):
    print('获取训练-测试数据中......')
    l = len(data)
    x = []
    y = []
    for i in range(l):
        temp = []
        y.append(data.iloc[i][0])
        for word in data.iloc[i][1].split():
            temp.append(vocab[word])
        x.append(temp)
        if i % 10000==0:
            print('%d句子已经处理' % (i))
    return x, np.array(y)

def pad_seq(x, maxlen):
    print('填充句子中......')
    data = np.zeros((len(x), maxlen))
    for i in range(len(x)):
        l = len(x[i])
        if l < 15:
            for j in range(l):
                data[i, j] = x[i][j]
        else:
            for j in range(maxlen):
                data[i, j] = x[i][j]
    return data

def split_data(x,y,bili):
    train = int(len(y) * bili)
    test = int(len(x) - train)

    train_x = x[0:train]
    train_y = y[0:train]
    test_x = x[-test:]
    test_y = y[-test:]
    return train_x, train_y, test_x, test_y

def caculate_acc(x, y):
    temp = np.exp(x)
    temp = temp / temp.sum(axis=1, keepdims=True)
    pre = np.argmax(temp, axis=1)
    right_pre = pre - y
    return len(right_pre[right_pre == 0]) / len(y)


def caculate_test_acc(x, y):
    temp = np.exp(x)
    temp = temp / temp.sum(axis=1, keepdims=True)
    pre = np.argmax(temp, axis=1)
    right_pre = pre - y
    return len(right_pre[right_pre == 0])

def get_pretrained_embed(path):
    w2v = Word2Vec.load(path)
    return len(w2v.wv.vectors), w2v.wv.vectors