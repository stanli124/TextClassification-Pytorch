'''
AUTHOR :li peng cheng

DATE :2021/07/23 17:05
'''
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class Text_RNN(torch.nn.Module):
    def __init__(self,
                 hidden_size=128,
                 seq_len=400,
                 class_num=1,
                 net_layer=1,
                 vocab_size=5000,
                 embed_dims=50,
                 batch_size=32,
                 device=None
                 ):
        super(Text_RNN, self).__init__()
        self.vocab_size=vocab_size
        self.hidden_size=hidden_size
        self.seq_len=seq_len
        self.class_num=class_num
        self.net_layer=net_layer
        self.embed_dims=embed_dims
        self.batch_size=batch_size
        self.device=device

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dims)
        self.lstm = nn.LSTM(input_size=self.embed_dims, hidden_size=self.hidden_size,batch_first=True)
        self.h0 = torch.randn((self.net_layer, self.batch_size, self.hidden_size)).to(self.device)
        self.c0 = torch.randn((self.net_layer, self.batch_size, self.hidden_size)).to(self.device)


        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.hidden_size, class_num)

    def forward(self, x):
        input = x.to(self.device)
        embedd = self.embedding(input)
        
        _, (hn, cn) = self.lstm(embedd, (self.h0, self.c0))
        # self.h0 = hn
        # self.c0 = cn
        # _,(hn,cn) = self.lstm(embedd)
        hn = hn.squeeze(0)
        out = self.fc(hn)
        out = self.sigmoid(out)
        return out