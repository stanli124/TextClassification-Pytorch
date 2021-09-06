'''
AUTHOR :li peng cheng

DATE :2021/08/18 21:31
'''
import numpy as np
import random
from gensim.models import Word2Vec

def get_Corpus(data):
    l = len(data)
    sentences = []
    for i in range(l):
        sentences.append(data[i].split(',')[1].strip().split())
        if i % 10000 ==0:
            print('已经处理%d' % (i))
    return sentences



if __name__ == '__main__':
    with open('fenci.txt','r', encoding='utf-8') as f:
        corpus = f.readlines()

    sentences = get_Corpus(corpus[0:200000])
    print('start training...')
    w2v = Word2Vec(sentences=sentences, size=50, window=5, min_count=0, iter=8, sg=0)
    w2v.save('./embedding_model/word2vec_model_8iter.model')
    np.save('wordvec_8iter_cbow_50dims', w2v.wv.vectors)

