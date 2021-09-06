'''
AUTHOR :li peng cheng

DATE :2021/08/10 21:58
'''
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import  torch.optim as optim
import torchtext
import torch
from collections import OrderedDict, Counter
from handle_text import *
from text_att_birnn import text_att_birnn
from gensim.models import Word2Vec

maxlen = 15
batch_size = 128
vocab_size = None
hidden_size = 128
class_num = 17
bidirection = True
embed_size = 50
epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# vocab = pd.read_csv('train_20wan.txt',header=None) #dataframe
# train = Counter(OrderedDict(vocab.values))   #dict
# vocab = torchtext.vocab.Vocab(train) #词典
w2v = Word2Vec.load('./embedding_model/word2vec_model_8iter_sg1.model')
print('model load success')

train = pd.read_csv('fenci.txt', header=None)[0:160000] #训练句子 最长的句子有54个词
print('fenci.txt load success')
# len_distribution = Counter(train.apply(lambda x:len(x.iloc[1].split()),axis=1))

# train_x, train_y = get_sequence(vocab, train)  #train_x由np.array组成的列表，train_y是真实类别
# train_x, train_y = get_and_pad_sequence(w2v, train, maxlen)  #train_x由np.array组成的列表，train_y是真实类别
train_x, train_y = get_pretrain_pad_seq(w2v.wv.index2word, train, maxlen)#获得索引
train_y = np.array(train_y)
# train_x = pad_sequence(train_x, maxlen)
train_x, train_y, test_x, test_y = split_data(train_x, train_y, 0.8)

#加载训练集和测试集
train_dataset = TensorDataset(torch.LongTensor(train_x), torch.LongTensor(train_y))
test_dataset = TensorDataset(torch.LongTensor(test_x), torch.LongTensor(test_y))
# train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
# test_dataset = TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

pretrain = get_pretrain_embedding()  #numpy
model = text_att_birnn(batch_size, embed_size, class_num,
                       bidirection, hidden_size, maxlen, len(w2v.wv.vocab), device, torch.tensor(pretrain))
model= model.to(device)

optim = optim.Adam(model.parameters(), lr=0.002,weight_decay=0.01)

loss = torch.nn.CrossEntropyLoss(reduction='sum',weight=torch.tensor([1.0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1]).to(device))

print(model.state_dict())

print('train_x shape：', train_x.shape) #(80000, 15)
print('train_y shape：', train_y.shape) #(80000, 15)
print('test_x shape：', test_x.shape)
print('test_y shape：', test_y.shape)

print('start training....')
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_acc = 0
    b = int(len(train_x) / batch_size)
    for iter, (x, y) in enumerate(train_loader):
        # print(x,y)
        optim.zero_grad()
        out = model(x.to(device))
        l = loss(out, y.to(device))
        l.backward()
        optim.step()
        total_loss += l.data
        acc = caculate_acc(out.cpu().detach().numpy(), y.numpy())
        total_acc += acc
        if iter % 50 ==0:
            print('Epoch %d. iter %d. loss is %f acc is %f' % (epoch, iter, l.data, acc))
    print('---------------Epoch %d. total loss is %f, total acc is %f----------------' % (epoch, total_loss, total_acc/b))

    test_acc = 0
    test_loss = 0
    model.eval()
    b = len(test_x)
    for iter, (x, y) in enumerate(test_loader):
        # print(x,y)
        out = model(x.to(device))
        l = loss(out, y.to(device))
        test_loss += l.data
        test_acc += caculate_test_acc(out.cpu().detach().numpy(), y.numpy())
    print('---------------Test dataset epoch %d. total loss is %f, total acc is %f----------------' % (epoch, test_loss, test_acc/b))
