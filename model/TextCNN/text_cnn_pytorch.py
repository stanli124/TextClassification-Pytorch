'''
AUTHOR :li peng cheng

DATE :2021/07/18 10:22
'''
import torch.nn as nn
import torch
import numpy as np


class TextCNN_torch(nn.Module):
    def __init__(self, vocab_size=None, embedding_dims=50,
                 seq_len=400,
                 kernel_size=[3, 4, 5],
                 out_channel=128,
                 device=None):
        super(TextCNN_torch,self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.seq_len = seq_len
        self.kernel_sizes = kernel_size
        self.class_num = 1
        self.out_channel = out_channel
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dims, padding_idx=0)
        self.conv1D_3 = nn.Conv1d(self.embedding_dims, self.out_channel, kernel_size[0])
        self.conv1D_4 = nn.Conv1d(self.embedding_dims, self.out_channel, kernel_size[1])
        self.conv1D_5 = nn.Conv1d(self.embedding_dims, self.out_channel, kernel_size[2])
        self.conv1D_6 = nn.Conv1d(self.embedding_dims, self.out_channel, kernel_size[3])
        self.global_max_pool_3 = nn.MaxPool1d(seq_len-kernel_size[0]+1, stride=1)
        self.global_max_pool_4 = nn.MaxPool1d(seq_len-kernel_size[1]+1, stride=1)
        self.global_max_pool_5 = nn.MaxPool1d(seq_len-kernel_size[2]+1, stride=1)
        self.global_max_pool_6 = nn.MaxPool1d(seq_len-kernel_size[3]+1, stride=1)
        self.fc = nn.Linear(in_features=out_channel*len(self.kernel_sizes), out_features=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(self.device)
        x = self.embedding(x)
        input = x.transpose(1, 2)

        out_3 = self.conv1D_3(input)
        out_3 = self.relu(out_3)
        out_3 = self.global_max_pool_3(out_3)

        out_4 = self.conv1D_4(input)
        out_4 = self.relu(out_4)
        out_4 = self.global_max_pool_4(out_4)

        out_5 = self.conv1D_5(input)
        out_5 = self.relu(out_5)
        out_5 = self.global_max_pool_5(out_5)

        out_6 = self.conv1D_6(input)
        out_6 = self.relu(out_6)
        out_6 = self.global_max_pool_6(out_6)
        
        out = torch.cat([out_3, out_4, out_5, out_6], dim=1)
        out = out.squeeze()
        out = self.fc(out)
        out = self.sigmoid(out)

        return out



