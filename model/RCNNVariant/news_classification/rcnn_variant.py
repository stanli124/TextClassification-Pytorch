'''
AUTHOR :li peng cheng

DATE :2021/09/06 9:05
'''
import torch
import torch.nn as nn

class rcnn_variant(nn.Module):
    def __init__(self, batch_size, embed_size, vocab_size, hidden_size, class_num,
                 pretrained_embed=None, device=None, pretrained=False):
        super(rcnn_variant, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.class_num = class_num
        self.pretrained = pretrained
        self.device = device
        self.pretrained_embed = pretrained_embed
        self.kernel_size = [1, 2, 3, 4, 5]

        if self.pretrained:
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.embed.from_pretrained(self.pretrained_embed)
        else:
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        self.bilstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, bidirectional=True)

        self.conv1d = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=self.embed_size+(2*self.hidden_size),
                                                             out_channels=128,
                                                             kernel_size=ks),
                                                   nn.ReLU()) for ks in self.kernel_size])

        self.Maxpool = nn.ModuleList([nn.MaxPool1d(kernel_size=15-ks+1) for ks in self.kernel_size])
        self.Avgpool = nn.ModuleList([nn.AvgPool1d(kernel_size=15-ks+1) for ks in self.kernel_size])

        self.classifier = nn.Linear(128*10, self.class_num)

    def forward(self, x):
        # x_left = torch.cat((x[:, 0:1], x[:, 0:-1]), dim=1)
        # x_right = torch.cat((x[:, 1:], x[:, -1:]), dim=1)

        x = self.embed(x) #torch.Size([128, 15, 50])
        x_lr, _ = self.bilstm(x) #torch.Size([128, 15, 256])
        input = torch.cat((x, x_lr), dim=2)  #[128, 15, 306]
        input = input.transpose(1, 2)   #[128, 306,15]

        conv_out = [conv(input) for conv in self.conv1d] #torch.Size([128, 128, 15]) torch.Size([128, 128, 14]) torch.Size([128, 128, 13]) torch.Size([128, 128, 12]) torch.Size([128, 128, 11])
        maxp_out = [self.Maxpool[i](conv_out[i]) for i in range(len(conv_out))] #维度都是torch.Size([128, 128, 1])
        avgp_out = [self.Avgpool[i](conv_out[i]) for i in range(len(conv_out))] #维度都是torch.Size([128, 128, 1])

        out = torch.cat(maxp_out + avgp_out, dim=1) #torch.Size([128, 1280, 1])
        out = out.reshape((self.batch_size, -1)) #torch.Size([128, 1280])
        out = self.classifier(out)  #torch.Size([128, 17])

        return out
