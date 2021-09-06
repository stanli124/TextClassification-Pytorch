'''
AUTHOR :li peng cheng

DATE :2021/07/18 10:21
'''
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch

def pad_Data_tensor(data):
    seq = torch.zeros(1,400)
    for i in range(len(data)):
        if len(data[i][0:400]) < 400:
            temp = torch.FloatTensor(data[i][0:400]).unsqueeze(0)
            temp = F.pad(temp, pad=(0, 400 - len(data[i][0:400])), value=0)
        else:
            temp = torch.FloatTensor(data[i][0:400]).unsqueeze(0)
        seq = torch.cat((seq,temp),0)
    return seq[1:]

def pad_Data_numpy(data):
    seq = np.zeros((1,400))
    for i in range(len(data)):
        length = len(data[i][0:400])
        if length < 400:
            temp = np.array(data[i][0:400]).reshape(1, length)
            temp = np.append(temp, np.zeros((1, 400-length)))
        else:
            temp = np.array(data[i][0:400]).reshape(1,400)
        seq = np.vstack((seq,temp))

    return seq[1:]

train_npz = np.load('train.npz',allow_pickle=True)
test_npz = np.load('test.npz',allow_pickle=True)
x_train, y_train = train_npz['x_train'], train_npz['y_train']
x_test, y_test = test_npz['x_test'], test_npz['y_test']

x_train = pad_Data_tensor(x_train)
x_test = pad_Data_tensor(x_test)

np.savez('dataset.npz', x_train=x_train.numpy(), y_train=y_train.reshape(1,25000), x_test=x_test.numpy(), y_test=y_test.reshape(1,25000))


