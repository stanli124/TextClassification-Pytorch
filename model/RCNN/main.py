import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

from rcnn import RCNN

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 10

print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# data = np.load('../TextCNN/dataset.npz')  #不知道是自己处理的数据有问题还是其它问题，模型训练的时候loss没有下降
# x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
# x_train[np.where(x_train == -1)] = 0
# x_test[np.where(x_test == -1)] = 0
# y_train[np.where(y_train == -1)] = 0
# y_test[np.where(y_test == -1)] = 0  # 把-1处理为0
# y_train = y_train.reshape(25000,1)
# y_test = y_test.reshape(25000,1)

train, test = np.load('../TextRNN/xunlian.npz', allow_pickle=True), np.load('../TextRNN/ceshi.npz', allow_pickle=True)
x_train, y_train = train['x_train'], train['y_train']
x_test, y_test = test['x_test'], test['y_test']

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen) #25000,400
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)  #25000,400
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Prepare input for model...')
x_train_current = x_train
x_train_left = np.hstack([np.expand_dims(x_train[:, 0], axis=1), x_train[:, 0:-1]]) #有两个0列，没有最后一列；x_train[:,0]取到第0列的数据；x_train[:, 0:-1]到0-398列
x_train_right = np.hstack([x_train[:, 1:], np.expand_dims(x_train[:, -1], axis=1)]) #有两个最后列，没有第0列；第1列到最后一列；最后一列
x_test_current = x_test
x_test_left = np.hstack([np.expand_dims(x_test[:, 0], axis=1), x_test[:, 0:-1]])
x_test_right = np.hstack([x_test[:, 1:], np.expand_dims(x_test[:, -1], axis=1)])
print('x_train_current shape:', x_train_current.shape)
print('x_train_left shape:', x_train_left.shape)
print('x_train_right shape:', x_train_right.shape)
print('x_test_current shape:', x_test_current.shape)
print('x_test_left shape:', x_test_left.shape)
print('x_test_right shape:', x_test_right.shape) #(25000, 400)

print('Build model...')
model = RCNN(maxlen, max_features, embedding_dims)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
print('Train...')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
model.fit([x_train_current, x_train_left, x_train_right], y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=([x_test_current, x_test_left, x_test_right], y_test))

model.summary()
print('Test...')
result = model.predict([x_test_current, x_test_left, x_test_right])
