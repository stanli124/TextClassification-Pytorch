'''
AUTHOR :li peng cheng

DATE :2021/08/23 17:57
'''
import torch
import torch.nn as nn

class rcnn(nn.Module):
    def __init__(self, batch_size, embed_size, vocab_size, hidden_size, class_num,
                 pretrained_embed=None, device=None, pretrained=False):
        super(rcnn, self).__init__()
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

        self.lstm_forward = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, bidirectional=True)
        # self.lstm_backward = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)
        self.conv1d = nn.Conv1d((self.hidden_size * 2 + self.embed_size), 64, kernel_size=1)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool1d(15-1+1, stride=1) #nn.MaxPool1d(kernel_size=1)
        # self.maxpool = nn.MaxPool1d(5, stride=5) #nn.MaxPool1d(kernel_size=1) #采样3个值
        self.classifier = nn.Linear(64, class_num)
        # self.classifier = nn.Linear(64*3, class_num)

    def forward(self, x):
        # x_left = torch.cat((x[:, 0:1], x[:, 0:-1]), dim=1)
        # x_right = torch.cat((x[:, 1:], x[:, -1:]), dim=1)

        x = self.embed(x) #torch.Size([128, 15, 50])
        x_lr, _ = self.lstm_forward(x) #torch.Size([128, 15, 256])
        out = torch.cat((x, x_lr), dim=2) #[128, 15, 306]
        out = out.transpose(1, 2)  #[128, 306,15]
        out = self.conv1d(out)  #[128, 64, 15]
        out = self.tanh(out)
        out = self.maxpool(out) #torch.Size([128, 64, 1])
        # out = out.squeeze()
        out = out.reshape((self.batch_size,-1))
        out = self.classifier(out)

        return out
