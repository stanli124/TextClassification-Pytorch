from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer

#attention继承子layer类，需要自己实现__init__,build,call三个函数。
#在第一次调用call时，会首先调用build函数
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim  #400
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # print('input shape: ',input_shape) #(None, 400, 256)
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        # print('W shape: ',self.W.shape) #(256,)
        self.features_dim = input_shape[-1] #256
        # print('features_dim shape: ',self.features_dim)

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # print(x.shape)
        features_dim = self.features_dim #256
        step_dim = self.step_dim         #400

        #x被reshape为[12800, 256]，W被reshape为[256,1] k.dot返回一个标量[12800,1]
        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        #e的维度[32,400] 32个句子，每个句子400个单词的注意力值

        # print('e shape',e.shape)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e) #a的维度[32,400]
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        #cast函数将张量转换为不同的dtype并返回
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())#a的维度[32,400] 这里句子中的400个单词对应的注意力权重已经求得；每个单词的值除以所有单词的和
        a = K.expand_dims(a)    #TensorShape([32, 400, 1])

        c = K.sum(a * x, axis=1) #c的维度[32,256]; 这一步求和是把每个句子的400个单词向量相加
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
