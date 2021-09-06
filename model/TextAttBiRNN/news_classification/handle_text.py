'''
AUTHOR :li peng cheng

DATE :2021/08/08 22:34
'''
from math import ceil
import numpy as np
import pandas as pd
# import jieba
from collections import Counter, OrderedDict


# 读取文件
# python自带的open方法
# with open('toutiao_cat_data.txt','r',encoding='utf-8') as f:
#     text = f.readlines()

# def get_class_sentence(text, stop_words):
#     text = text.apply(lambda x: x.iloc[0].split('_!_')[1:4],axis=1)
#     for i in range(len(text)):
#         text[i][0] = int(text[i][0])-100
#         outstr = ''
#         for word in jieba.cut(text[i][2].strip()):
#             if word not in stop_words:
#                 outstr+=word+' '
#         text[i][2] = outstr.strip(' ')
#         if i % 1000 == 0:
#             print('已经处理到%d行：%s' % (i, outstr))
#
#     category = []
#     sentence = []
#     for i in range(len(text)):
#         category.append(text[i][0])
#         sentence.append(text[i][2])
#     return category,sentence


def get_vocab(fenci):
    fenci = fenci[0:200000]
    outstr = fenci.iloc[0][1]
    for i in range(len(fenci) - 1):
        outstr = outstr + ' ' + fenci.iloc[i + 1][1]
        if i % 1000 == 0:
            print('已经处理到%d行' % (i))

    return Counter(outstr.split())  # 将20万个句子分开，并统计每个词的词频


def get_sequence(vocab, data):
    y = []
    x = []
    for i in range(len(data)):
        temp = []
        y.append(data.iloc[i, 0])
        for word in data.iloc[i, 1].split():
            temp.append(vocab[word])
        x.append(np.array(temp))
        # if i % 5000 ==0:
        #     print('%d句子正在处理' % (i))
    return x, y


def pad_sequence(data, maxlen):
    back_data = np.ones((len(data), maxlen))
    for i in range(len(back_data)):
        if i % 1000 == 0:
            print('%d句子已处理' % (i))
        l = len(data[i])
        if l < 15:
            for j in range(l):
                back_data[i, j] = data[i][j]
        else:
            for j in range(maxlen):
                back_data[i, j] = data[i][j]
    return back_data


def split_data(x, y, train_per=0.8):
    size = len(y)
    train_size = ceil(size * train_per)
    test_size = int(size - train_size)
    return x[0:train_size], y[0:train_size], x[-test_size:], y[-test_size:]


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


def get_and_pad_sequence(vocab, data, maxlen):
    y = []
    x = []
    for i in range(len(data)):
        temp = []
        y.append(data.iloc[i, 0])
        for word in data.iloc[i, 1].split():
            # print(i)
            if word in [',', 'Yes', '习凿', 'Tesla', '有令', '237', '西泠', '从苏', '兵坛', '登山者', '梅毒', '只闻',
                        'YOTA3', '粤绣', '艾美人', '相知', '雪天', '帮爸', '内外交困', 'vacation', '这名常', 'Instead',
                        '长江日报', 'but', '数十名']:
                break
            temp.append(vocab[word])

        x.append(temp)
        # if i % 5000 ==0:
        #     print('%d句子正在处理' % (i))
    zero = np.zeros(50)
    for i in range(len(x)):
        l = len(x[i])
        if l < maxlen:
            for j in range(maxlen - l):
                x[i].append(zero)
        else:
            x[i] = x[i][0:maxlen]

    return np.array(x), y


def get_pretrain_embedding():
    a = np.load('wordvec_8iter_sk-gm_50dims.npy')
    b = np.zeros(50).reshape(1, 50)
    return np.append(b, a, axis=0)


def get_pretrain_pad_seq(vocab, data, maxlen):
    y = []
    x = []
    for i in range(len(data)):
        temp = []
        y.append(data.iloc[i, 0])
        for word in data.iloc[i, 1].split():
            # print(i)
            if word in [',', 'Yes', '习凿', 'Tesla', '有令', '237', '西泠', '从苏', '兵坛', '登山者', '梅毒', '只闻',
                        'YOTA3', '粤绣', '艾美人', '相知', '雪天', '帮爸', '内外交困', 'vacation', '这名常', 'Instead',
                        '长江日报', 'but', '数十名']:
                break
            temp.append(vocab.index(word) + 1)
        if i % 10000 ==0:
            print('已经处理%d' % i)
        x.append(temp)

    temp = np.zeros((len(data), maxlen))
    for i in range(len(x)):
        l = len(x[i])
        if l < maxlen:
            for j in range(l):
                temp[i, j] = x[i][j]
        else:
            for j in range(maxlen):
                temp[i, j] = x[i][j]
        if i % 10000 ==0:
            print('已经处理%d' % i)
    return temp, y

# text = pd.read_table('./toutiao_cat_data.txt',encoding='utf-8',sep='\n',header=None)
# stop_words = list(pd.read_table('cn_stopwords.txt',encoding='utf-8',header=None)[0])
#
# fenci = pd.read_csv('fenci.txt',encoding='utf-8',header=None)
#
# vocab = get_vocab(fenci)  #每个词的词频
# vocab = pd.DataFrame({'word':list(vocab.keys()), 'freq':list(vocab.values())})
# vocab.sort_values(by='freq',ascending=False,inplace=True)
# vocab = OrderedDict(vocab.values)
