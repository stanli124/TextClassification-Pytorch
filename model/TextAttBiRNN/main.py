from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import numpy as np
from text_att_birnn import TextAttBiRNN

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 5

print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
train, test = np.load('../TextRNN/xunlian.npz',allow_pickle=True), np.load('../TextRNN/ceshi.npz', allow_pickle=True)
x_train, y_train = train['x_train'], train['y_train']
x_test, y_test = test['x_test'], test['y_test']
print(len(x_train), 'train sequences') #25000
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape) #25000，400
print('x_test shape:', x_test.shape)

print('Build model...')
model = TextAttBiRNN(maxlen, max_features, embedding_dims)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=(x_test, y_test))

print('Test...')
result = model.predict(x_test)
