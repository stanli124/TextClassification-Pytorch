from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM as LSTM
from attention import Attention


class TextAttBiRNN(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        super(TextAttBiRNN, self).__init__()
        self.maxlen = maxlen                 #序列长度；句子长度
        self.max_features = max_features     #词典大小
        self.embedding_dims = embedding_dims #嵌入的维度
        self.class_num = class_num           #分类类别
        self.last_activation = last_activation  #使用的机激活函数
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.bi_rnn = Bidirectional(LSTM(128, return_sequences=True))  # LSTM or GRU
        self.attention = Attention(self.maxlen)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextAttBiRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextAttBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs) #(None, 400, 50)
        x = self.bi_rnn(embedding) #(None, 400, 256)
        x = self.attention(x)      #(None, 256)
        output = self.classifier(x) #(None, 1)
        return output
