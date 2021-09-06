'''
AUTHOR :li peng cheng

DATE :2021/08/11 22:34
'''
import torch
import numpy as np
import torch.nn as nn

class text_att_birnn(nn.Module):
    def __init__(self,
                 batch_size,
                 embedding_size,
                 class_num,
                 bidirection=False,
                 hidden_size = 128,
                 max_seq = 15,
                 vocab_size = None,
                 device = None,
                 pretrain = None):
        super(text_att_birnn, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.class_num = class_num
        self.bidirection = bidirection
        self.max_seq = max_seq
        self.device = device
        self.pretrain = pretrain


        self.embedding = nn.Embedding(self.vocab_size+1, self.embedding_size)
        self.embedding.from_pretrained(self.pretrain)
        self.tanh = nn.Tanh()
        self.epsi = torch.tensor([0.0000001]).to(self.device)
        if self.bidirection:
            self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(2*self.hidden_size, self.class_num)
            self.softmax = nn.Softmax(dim=1)
            # self.h0 = torch.tensor(np.random.randn(2, self.batch_size, self.hidden_size),dtype=torch.float32).to(self.device)
            # self.c0 = torch.tensor(np.random.randn(2, self.batch_size, self.hidden_size),dtype=torch.float32).to(self.device)
            # self.h0 = torch.tensor(np.random.randn(2, self.batch_size, self.hidden_size)).to(self.device)
            # self.c0 = torch.tensor(np.random.randn(2, self.batch_size, self.hidden_size)).to(self.device)
            # self.h0 = torch.randn(2,self.batch_size,self.hidden_size).to(self.device)
            # self.c0 = torch.randn(2,self.batch_size,self.hidden_size).to(self.device)
            self.W = nn.Parameter(torch.tensor(np.random.randn(2*self.hidden_size), dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(np.random.randn(self.max_seq), dtype=torch.float32))

        else:
            self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=False, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, self.class_num)
            self.softmax = nn.Softmax()
            self.h0 = torch.tensor(np.random.randn(1, self.batch_size, self.hidden_size)).to(self.device)
            self.c0 = torch.tensor(np.random.randn(1, self.batch_size, self.hidden_size)).to(self.device)
            self.W = nn.Parameter(torch.tensor(np.random.randn(self.hidden_size), dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(np.random.randn(self.max_seq), dtype=torch.float32))


    def attention(self,x):
        if self.bidirection:
            features_dim = 2*self.hidden_size
            time_step = self.max_seq
            e = torch.mm(x.reshape(-1,features_dim), self.W.reshape(features_dim, 1))
            e = e.reshape(self.batch_size, self.max_seq)
            e = e + self.b
            e = self.tanh(e)

            a = torch.exp(e)
            a_ = a / a.sum(dim=1, keepdim=True)
            a_ = a_.unsqueeze(2)

            # a_ = torch.where(torch.isnan(a_), torch.full_like(a_, 0), a_)
            # x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)

            c = torch.sum(((a_ * x) + self.epsi), axis=1)
            # c = torch.where(torch.isnan(c), torch.full_like(c, 0), c)

        else:
            features_dim = self.hidden_size
            time_step = self.max_seq
            e = torch.mm(x.reshape(-1, features_dim), self.W.reshape(features_dim, 1))
            e = e.reshape(self.batch_size, self.max_seq)
            e = e + self.b
            e = self.tanh(e)

            a = torch.exp(e)
            a /= a.sum(dim=1, keepdim=True)
            a = a.unsqueeze(2)

            c = torch.sum((a * x)+self.epsi, axis=1)
        return c

    def forward(self, x):


        #out无nan值
        out = self.embedding(x)
        # print(out)
        # print('out shape: ',out.shape)

        #有nan值
        # output, (hn, cn) = self.lstm(out, (self.h0, self.c0))
        output, (hn, cn) = self.lstm(out)


        # hn = hn.permute(1, 0, 2).reshape((self.batch_size,-1))
        output = self.attention(output)


        output = self.fc(output)

        # output = self.softmax(output)
        return output

