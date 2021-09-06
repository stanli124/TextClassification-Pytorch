'''
AUTHOR :li peng cheng

DATE :2021/07/27 21:05
'''
import torch
import torch.nn as nn

# class Text_BiRNN(nn.Module):
#     def __init__(self,
#                  embedding_dims=50,
#                  hidden_size=128,
#                  num_layers=1,
#                  batch_size=32,
#                  vocab_size=5000,
#                  class_num = 2,
#                  device=None,
#                  bidirect=False):
#         super(Text_BiRNN, self).__init__()
#         self.embedding_dims = embedding_dims
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.bidirect = bidirect
#         self.vocab_size = vocab_size
#         self.class_num = class_num
#         self.device = device
#
#         if self.bidirect==True:
#             self.num_direc = 2
#         self.embedding = nn.Embedding(self.vocab_size, self.embedding_dims)
#         self.bi_lstm = nn.LSTM(self.embedding_dims, self.hidden_size, self.num_layers, batch_first=True,bidirectional=self.bidirect)
#         self.fc1 = nn.Linear(self.num_direc*self.hidden_size, self.hidden_size)
#         # self.fc1 = nn.Linear(self.num_direc*self.hidden_size, self.class_num)
#         self.fc2 = nn.Linear(self.hidden_size, self.class_num)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax()
#         self.sigmoid = nn.Sigmoid()
#         self.h0 = torch.randn((self.num_layers*self.num_direc, self.batch_size,self.hidden_size)).cuda()
#         self.c0 = torch.randn((self.num_layers*self.num_direc, self.batch_size,self.hidden_size)).cuda()
#
#     def forward(self, input):
#         x = input.to(self.device)
#         x = self.embedding(x)
#         out, (hn,cn) = self.bi_lstm(x,(self.h0,self.c0))
#         hn = hn.permute(1,0,2).reshape((self.batch_size,-1))
#         out = self.fc1(hn)
#         out = self.relu(out)
#         out = self.fc2(out)
#         # out = self.softmax(out)
#         out = self.sigmoid(out)
#
#
#
#         # print(out.shape)
#         return out




class Text_BiRNN(nn.Module):
    def __init__(self,
                 batch_size=32,
                 embedding_dims=50,
                 class_num=2,
                 bidirect=False,
                 hidden_size=128,
                 max_len = None,
                 vocab_size=None,
                 device=None
                            ):
        super(Text_BiRNN, self).__init__()
        self.embedding_dims = embedding_dims
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bidirect = bidirect
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.device = device

        if self.bidirect==True:
            self.num_direc = 2
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dims)
        self.bi_lstm = nn.LSTM(self.embedding_dims, self.hidden_size, 1, batch_first=True,bidirectional=self.bidirect)
        self.fc1 = nn.Linear(self.num_direc*self.hidden_size, self.hidden_size)
        # self.fc1 = nn.Linear(self.num_direc*self.hidden_size, self.class_num)
        self.fc2 = nn.Linear(self.hidden_size, self.class_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.h0 = torch.randn((1*self.num_direc, self.batch_size,self.hidden_size)).cuda()
        self.c0 = torch.randn((1*self.num_direc, self.batch_size,self.hidden_size)).cuda()

    def forward(self, input):
        # x = input.to(self.device)
        x = self.embedding(input)
        out, (hn, cn) = self.bi_lstm(x,(self.h0,self.c0))
        hn = hn.permute(1,0,2).reshape((self.batch_size, -1))
        out = self.fc1(hn)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.softmax(out)
        # out = self.sigmoid(out)



        # print(out.shape)
        return out


