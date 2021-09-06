'''
AUTHOR :li peng cheng

DATE :2021/07/18 10:21
'''
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from text_cnn_pytorch import TextCNN_torch


def float2int(out):
    out[out > 0.5] = 1
    out[out < 0.5] = 0
    return out

def caculate_acc(pre, real):
    n = len(real)
    temp = np.abs(pre - real).ravel().sum()
    out = temp / float(n)
    return out

embedding_dims = 50
vocab_size = 5000
seq_len = 400
kernel_size = [3, 4, 5, 6]
out_channel = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('load data....')
data = np.load('dataset.npz')
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
x_train[np.where(x_train == -1)] = 0
x_test[np.where(x_test == -1)] = 0
y_train[np.where(y_train == -1)] = 0
y_test[np.where(y_test == -1)] = 0  # 把-1处理为0

x_train = torch.LongTensor(x_train)  # 25000,400
y_train = torch.FloatTensor(y_train.T)  # 25000,1
x_test = torch.LongTensor(x_test)  # 25000,400
y_test = torch.FloatTensor(y_test.T)  # 25000,1

x_train = torch.cat([x_train, x_test[0:15000]], dim=0)
y_train = torch.cat([y_train, y_test[0:15000]], dim=0)

train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

print('bulid model....')
model = TextCNN_torch(vocab_size, embedding_dims, seq_len, kernel_size, out_channel, device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss = torch.nn.BCELoss()

samples = x_test[-5000:]
real = y_test[-5000:].to(device)

print('start train.....')
for epoch in range(5):
    model.train()
    for index, (x_batch, y_batch) in enumerate(train_dataloader):
        output = model(x_batch)
        l = loss(output, y_batch.to(device))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if index % 50 == 0:
            print('epoch %d , iter is %d , loss is %f' % (epoch, index, l.data))
        # print(index)
        # print(x_batch.shape, y_batch.shape)
        # break
    if epoch % 2 == 0:
        model.eval()
        out = model(samples)  # 预测值
        l = loss(out, real)  # 预测和真实值的损失值
        pre = float2int(out)
        acc = caculate_acc(pre.detach().cpu().numpy(), real.detach().cpu().numpy())
        print('epoch %d test(5000 samples) loss is %f acc is %f' % (epoch, l.data, (1 - acc) * 100))


model.eval()
out = model(x_test[-10000:])  # 预测值
l = loss(out, y_test[-10000:].to(device))  # 预测和真实值的损失值

pre = float2int(out)
real = y_test[-10000:].to(device)
acc = caculate_acc(pre.detach().cpu().numpy(), real.detach().cpu().numpy())

print('Final test(10000 samples) loss is %f' % (l.data))
print('Final test(10000 samples) acc is %f' % ((1 - acc) * 100))
