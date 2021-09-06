'''
AUTHOR :li peng cheng

DATE :2021/08/23 17:58
'''
import numpy as np
import random
import pandas as pd
from gensim.models import Word2Vec

def get_Corpus(data):
    l = len(data)
    sentences = []
    for i in range(l):
        # sentences.append(data[i].split(',')[1].strip().split())
        sentences.append(data.iloc[i][1].strip().split())
        if i % 10000 ==0:
            print('已经处理%d' % (i))
    return sentences



if __name__ == '__main__':
    # with open('../data/fenci.txt', 'r', encoding='utf-8') as f:
    #     corpus = f.readlines()
    #
    # sentences = get_Corpus(corpus[0:160000])

    corpus = pd.read_csv('../data/fenci.txt', header=None)[0:160000]
    sentences = get_Corpus(corpus[0:160000])

    print('start training...')
    w2v = Word2Vec(sentences=sentences, size=100, window=5, min_count=1, iter=8, sg=1)
    w2v.save('w2v_model_8iter_skgr_100dim.model')
    # np.save('wordvec_8iter_cbow_50dims', w2v.wv.vectors)