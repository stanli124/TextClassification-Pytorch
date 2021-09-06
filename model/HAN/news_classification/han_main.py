'''
AUTHOR :li peng cheng

DATE :2021/09/06 16:01
'''
from utils import *
from torch.utils.data import TensorDataset, DataLoader
# from torchsummary import summary
from han import han
import pandas as pd
import torch
import torch.optim as optim
import numpy as np


batch_size = 128
maxlen = 15
vocab_size = None
hidden_size = 128
class_num = 17
pretrained_embed = None
embed_size = 50
epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained = True

corpus = pd.read_csv('./data/fenci.txt', header=None)[0:160000]
vocab = get_vocab(corpus)  #得到词典
vocab_size = len(vocab)    #16万个句子    107788包含unk和pad两个

x, y = obtain_data(vocab, corpus)
print(len(x))
print(len(y))

x = pad_seq(x, maxlen) #(size, maxlen)

train_x, train_y, test_x, test_y = split_data(x, y, 0.8)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

train_dataset = TensorDataset(torch.LongTensor(train_x), torch.LongTensor(train_y))
test_dataset = TensorDataset(torch.LongTensor(test_x), torch.LongTensor(test_y))

print('train_x.shape', train_dataset.tensors[0].shape)
print('train_y.shape', train_dataset.tensors[1].shape)
print('test_x.shape', test_dataset.tensors[0].shape)
print('test_y.shape', test_dataset.tensors[1].shape)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

if pretrained:
    path = './embedding_model/w2v_model_8iter_skgr_50dim.model'
    vocab_size, pretrained_embed = get_pretrained_embed(path)
    pretrained_embed = np.append(np.zeros((2, embed_size)), pretrained_embed, axis=0)
    model = han(batch_size, embed_size, vocab_size+2, hidden_size, class_num, torch.tensor(pretrained_embed), device, pretrained)
else:
    model = han(batch_size, embed_size, vocab_size, hidden_size, class_num, pretrained_embed, device, pretrained)

model = model.to(device)

optim = optim.Adam(model.parameters(), lr=0.002,weight_decay=0.01)
loss = torch.nn.CrossEntropyLoss(reduction='sum',weight=torch.tensor([1.0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1]).to(device))
print(model)

print('start training......')
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
