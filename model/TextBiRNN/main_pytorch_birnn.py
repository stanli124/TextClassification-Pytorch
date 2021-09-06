# '''
# AUTHOR :li peng cheng
#
# DATE :2021/07/27 21:05
# '''
# import torch
# import numpy as np
# import torch.nn as nn
# from torch.utils.data import TensorDataset,DataLoader,Dataset
# from text_birnn_pytorch import Text_BiRNN
#
# def handle_data(data, seq_len):
#     l = len(data)
#     max_len = seq_len
#     temp = np.zeros((l, max_len))
#
#     for i in range(l):
#         juzi = len(data[i])
#         if juzi < 400:
#             for  j in range(juzi):
#                 temp[i,j] = data[i][j]
#         if juzi >= 400:
#             for j in range(max_len):
#                 temp[i,j] = data[i][j]
#
#     return temp
#
# def caculate_acc(x,y):
#     return np.abs(x-y).sum()
#
#
#
# hidden_size = 128
# embeddings_dims = 60
# batch_size = 32
# class_num = 1
# net_layers = 1
# vocab_size = 5000
# bidirection = True
# devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# max_len = 400
#
# epochs = 10
#
# train, test = np.load('../TextRNN/xunlian.npz', allow_pickle=True), np.load('../TextRNN/ceshi.npz', allow_pickle=True)
# x_train, y_train = train['x_train'], train['y_train']
# x_test, y_test = test['x_test'], test['y_test']
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
#
# x_train = handle_data(x_train, max_len)
# x_test = handle_data(x_test, max_len)
#
# x_train = torch.LongTensor(x_train)
# y_train = torch.FloatTensor(y_train).unsqueeze(1)
# x_test = torch.LongTensor(x_test)
# y_test = torch.FloatTensor(y_test).unsqueeze(1)
#
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
#
# train_dataset = TensorDataset(x_train, y_train)
# train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
# test_dataset = TensorDataset(x_test, y_test)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#
# model = Text_BiRNN(embeddings_dims,hidden_size,net_layers,batch_size,vocab_size,class_num,devices,bidirection)
# model = model.cuda()
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
# metric = nn.BCELoss()
#
# print('start training...')
# for i in range(epochs):
#     acc = 0
#     for idx, (x,y) in enumerate(train_dataloader):
#         x = x.to(devices)
#         y = y.to(devices)
#         optimizer.zero_grad()
#         out = model(x)
#         l = metric(out, y)
#         l.backward()
#         optimizer.step()
#         if idx % 50 ==0:
#             print('Epoch %d iter %d loss is %f' % (i, idx, l.data))
#     for idx, (x,y) in enumerate(test_dataloader):
#         x = x.to(devices)
#         out = model(x)
#         # test_out.append(torch.round(out).detach().cpu().numpy().reshape((batch_size,1)))
#         acc += caculate_acc(torch.round(out).detach().cpu().numpy().reshape((batch_size,1)), y.numpy())
#     print('Epoch %d accuracy is %f' % (i, (25000-acc)/25000) )
#
#



'''
AUTHOR :li peng cheng

DATE :2021/08/10 21:58
'''
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
import  torch.optim as optim
import torchtext
import torch
from collections import OrderedDict, Counter
from handle_text import *
from text_birnn_pytorch import Text_BiRNN

maxlen = 15
batch_size = 128
vocab_size = None
hidden_size = 128
class_num = 17
bidirection = True
embed_size = 50
epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vocab = pd.read_csv('../TextAttBiRNN/news_classification/train.txt',header=None) #dataframe
train = Counter(OrderedDict(vocab.values))   #dict
vocab = torchtext.vocab.Vocab(train) #词典

train = pd.read_csv('../TextAttBiRNN/news_classification/fenci.txt', header=None)[0:40000] #训练句子 最长的句子有54个词
# len_distribution = Counter(train.apply(lambda x:len(x.iloc[1].split()),axis=1))

train_x, train_y = get_sequence(vocab, train)  #train_x由np.array组成的列表，train_y是真实类别
train_y = np.array(train_y)
train_x = pad_sequence(train_x, maxlen)
train_x, train_y, test_x, test_y = split_data(train_x, train_y, 0.8)

#加载训练集和测试集
train_dataset = TensorDataset(torch.LongTensor(train_x), torch.LongTensor(train_y))
test_dataset = TensorDataset(torch.LongTensor(test_x), torch.LongTensor(test_y))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = Text_BiRNN(batch_size,embed_size,class_num,bidirection,hidden_size,maxlen, len(vocab), device)
model= model.to(device)

optim = optim.Adam(model.parameters(), lr=0.002,weight_decay=0.05)

loss = torch.nn.CrossEntropyLoss(reduction='sum',weight=torch.tensor([1.0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1]).to(device))

print(model)

print('train_x shape：', train_x.shape) #(80000, 15)
print('train_y shape：', train_y.shape) #(80000, 15)
print('test_x shape：', test_x.shape)
print('test_y shape：', test_y.shape)


print('start training...')
for i in range(epochs):
    acc = 0
    model.train()
    total_loss = 0
    for idx, (x,y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad()
        out = model(x)
        l = loss(out, y)
        l.backward()
        optim.step()
        total_loss+=l.data
        if idx % 50 ==0:
            print('Epoch %d iter %d loss is %f' % (i, idx, l.data))
    print('Epoch %d loss is %f' % (i, total_loss))

    # for idx, (x,y) in enumerate(test_dataloader):
    #     x = x.to(devices)
    #     out = model(x)
    #     # test_out.append(torch.round(out).detach().cpu().numpy().reshape((batch_size,1)))
    #     acc += caculate_acc(torch.round(out).detach().cpu().numpy().reshape((batch_size,1)), y.numpy())
    # print('Epoch %d accuracy is %f' % (i, (25000-acc)/25000) )



