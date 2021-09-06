from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import numpy as np
from text_birnn import TextBiRNN

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 10

print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# data = np.load('../TextCNN/dataset.npz')
# x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
# x_train[np.where(x_train == -1)] = 0
# x_test[np.where(x_test == -1)] = 0
# y_train[np.where(y_train == -1)] = 0
# y_test[np.where(y_test == -1)] = 0  # 把-1处理为0
# y_train = y_train.reshape(25000,1)
# y_test = y_test.reshape(25000,1)


train, test = np.load('../TextRNN/xunlian.npz',allow_pickle=True), np.load('../TextRNN/ceshi.npz', allow_pickle=True)
x_train, y_train = train['x_train'], train['y_train']
x_test, y_test = test['x_test'], test['y_test']

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

print('Build model...')
model = TextBiRNN(maxlen, max_features, embedding_dims)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=(x_test, y_test))

print('Test...')
result = model.predict(x_test)
