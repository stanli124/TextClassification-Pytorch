'''
AUTHOR :li peng cheng

DATE :2021/09/06 16:01
'''
import torch
import torch.nn as nn
import numpy as np


class han(nn.Module):
    def __init__(self, batch_size, embed_size, vocab_size, hidden_size, class_num,
                 pretrained_embed=None, device=None, pretrained=False):
        super(han, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.class_num = class_num
        self.pretrained = pretrained
        self.device = device
        self.pretrained_embed = pretrained_embed

        if self.pretrained:
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.embed.from_pretrained(self.pretrained_embed)
        else:
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        self.bilstm_1 = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.bilstm_2 = nn.LSTM(self.hidden_size * 2, self.hidden_size, batch_first=True, bidirectional=True)

        self.W1 = nn.Parameter(torch.tensor(np.random.randn(2 * self.hidden_size), dtype=torch.float32))
        self.b1 = nn.Parameter(torch.tensor(np.random.randn(15), dtype=torch.float32))
        self.W2 = nn.Parameter(torch.tensor(np.random.randn(2 * self.hidden_size), dtype=torch.float32))
        self.b2 = nn.Parameter(torch.tensor(np.random.randn(15), dtype=torch.float32))
        self.epsi = torch.tensor([0.0000001]).to(self.device)
        self.tanh = nn.Tanh()

        self.classifier = nn.Linear(128 * 2, self.class_num)

    def attention_1(self, x):
        features_dim = 2 * self.hidden_size

        e = torch.mm(x.reshape(-1, features_dim), self.W1.reshape(features_dim, 1))
        e = e.reshape(self.batch_size, 15)  #(batch_size,15)
        e = e + self.b1        #15个值，每个值都加在列上，每一列加的都是同一个值
        e = self.tanh(e)

        a = torch.exp(e)  #(batch_size,15)
        a_ = a / a.sum(dim=1, keepdim=True)  #(batch_size, 15) 得到每个单词的重要程度，这里总和为1
        a_ = a_.unsqueeze(2) #(batch_size,15,1)

        # a_ = torch.where(torch.isnan(a_), torch.full_like(a_, 0), a_)
        # x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)

        # c = torch.sum(((a_ * x) + self.epsi), axis=1) #(batch_size,256)
        c = ((a_ * x) + self.epsi) #(batch_size, 15, 256)
        return c

    def attention_2(self, x):
        features_dim = 2 * self.hidden_size

        e = torch.mm(x.reshape(-1, features_dim), self.W2.reshape(features_dim, 1))
        e = e.reshape(self.batch_size, 15)  #(batch_size,15)
        e = e + self.b2        #15个值，每个值都加在列上，每一列加的都是同一个值
        e = self.tanh(e)

        a = torch.exp(e)  #(batch_size,15)
        a_ = a / a.sum(dim=1, keepdim=True)  #(batch_size, 15)
        a_ = a_.unsqueeze(2) #(batch_size,15,1)

        # a_ = torch.where(torch.isnan(a_), torch.full_like(a_, 0), a_)
        # x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)

        c = torch.sum(((a_ * x) + self.epsi), axis=1) #(batch_size,256)
        # c = ((a_ * x) + self.epsi) #(batch_size, 15, 256)
        return c


    def forward(self, x):
        # x_left = torch.cat((x[:, 0:1], x[:, 0:-1]), dim=1)
        # x_right = torch.cat((x[:, 1:], x[:, -1:]), dim=1)

        x = self.embed(x)  # torch.Size([batch_size, 15, 50])

        out, _ = self.bilstm_1(x)  #torch.Size([batch_size, 15, 256])
        out = self.attention_1(out)  #(batch_size, 15, 256)

        out, _ = self.bilstm_2(out) #(batch_size, 15, 256)
        out = self.attention_2(out) #(batch_size, 256)

        # out = torch.sum((out + self.epsi), axis=1)

        out = self.classifier(out)  # torch.Size([batch_size, 17])

        return out
