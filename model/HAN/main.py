from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

from han import HAN
import numpy as np

max_features = 5000
maxlen_sentence = 16
maxlen_word = 25
batch_size = 32
embedding_dims = 50
epochs = 10

print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
train, test = np.load('../TextRNN/xunlian.npz', allow_pickle=True), np.load('../TextRNN/ceshi.npz', allow_pickle=True)
x_train, y_train = train['x_train'], train['y_train']
x_test, y_test = test['x_test'], test['y_test']

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x #sentence x #word)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen_sentence * maxlen_word)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen_sentence * maxlen_word)
x_train = x_train.reshape((len(x_train), maxlen_sentence, maxlen_word))
x_test = x_test.reshape((len(x_test), maxlen_sentence, maxlen_word))
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = HAN(maxlen_sentence, maxlen_word, max_features, embedding_dims)
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
