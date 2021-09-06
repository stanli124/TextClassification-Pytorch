'''
AUTHOR :li peng cheng

DATE :2021/07/23 17:05
'''
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from text_rnn_pytorch import Text_RNN

input_size = 50
hidden_size = 128
seq_len = 400
class_num = 1
net_layer = 1
vocab_size = 5000
batch_size = 32
embed_dim = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def float2int(out):
    out[out > 0.5] = 1
    out[out < 0.5] = 0
    return out


def caculate_acc(pre, real):
    n = len(real)
    temp = np.abs(pre - real).ravel().sum()
    out = temp / float(n)
    return out


def handle_data(data, seq_len):
    start = time.time()
    seq_lengths = []
    for seq in data:
        seq_lengths.append(len(seq))
    max_len = seq_len
    if max(seq_lengths) <= max_len:
        max_len = max(seq_lengths)
    temp = np.zeros((len(seq_lengths), max_len))

    seq_num = 0
    for seq in data:
        l = len(seq)
        if l < max_len:
            for i in range(l):
                temp[seq_num, i] = seq[i]
        else:
            for i in range(max_len):
                temp[seq_num, i] = seq[i]
        seq_num += 1
    end = time.time()
    # print('运行时间', end-start)
    return torch.LongTensor(temp), torch.tensor(seq_lengths)


print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
train, test = np.load('xunlian.npz', allow_pickle=True), np.load('ceshi.npz', allow_pickle=True)
x_train, y_train = train['x_train'], train['y_train']
x_test, y_test = test['x_test'], test['y_test']

# x_train = np.append(x_train, x_test[0:15000],axis=0)
# y_train = np.append(y_train, y_test[0:15000],axis=0)
# x_test = x_test[-10000:]
# y_test = y_test[-10000:]

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print()

x_train, train_seq_len = handle_data(x_train, seq_len)  # train_seq_len是每个句子的长度
x_test, test_seq_len = handle_data(x_test, seq_len)
y_train = torch.FloatTensor(y_train).reshape((len(y_train), 1))
y_test = torch.FloatTensor(y_test).reshape((len(y_test), 1))

print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)

train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = Text_RNN(hidden_size, seq_len, class_num, net_layer, vocab_size, embed_dim, batch_size, device)
model = model.to(device)

for p in model.parameters():
    print(p,p.numel())
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss = torch.nn.BCELoss()

samples = x_test[-5000:]
real = y_test[-5000:].to(device)
test_dataset = TensorDataset(samples, real)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print('start train.....')
for epoch in range(13):
    model.train()
    for index, (x_batch, y_batch) in enumerate(train_dataloader):
        output = model(x_batch)
        # print(output)
        l = loss(output, y_batch.to(device))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if index % 50 == 0:
            print('epoch %d , iter is %d , loss is %f' % (epoch, index, l.data))
        # print(index)
        # print(x_batch.shape, y_batch.shape)
        # break

    if epoch % 1 == 0:
        test_loss=[]
        test_acc=[]
        model.eval()
        for idx, (x, y) in enumerate(test_dataloader):
            out = model(x)  # 预测值
            l = loss(out, y)  # 预测和真实值的损失值
            test_loss.append(l.data)
            pre = float2int(out)
            acc = 1 - caculate_acc(pre.detach().cpu().numpy(), y.detach().cpu().numpy())
            test_acc.append(acc)
        print('epoch %d test(4992 samples) loss is %f acc is %f' % (epoch, sum(test_loss)/len(test_loss), sum(test_acc)/len(test_acc)))

# _, idx_sort_train = torch.sort(train_seq_len, descending=True) #按照句子长度降序排列
# _, idx_sort_test = torch.sort(test_seq_len, descending=True)
#
# x_train_packed = nn.utils.rnn.pack_padded_sequence(input=x_train.index_select(0, idx_sort_train), lengths=train_seq_len[idx_sort_train], batch_first=True)
# x_test_packed = nn.utils.rnn.pack_padded_sequence(x_test.index_select(0, idx_sort_test),test_seq_len[idx_sort_test],batch_first=True)
# print(x_train_packed)
# print(x_test_packed)
