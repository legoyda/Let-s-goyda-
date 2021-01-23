import tensorflow
import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
max_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
x_train[3]



y_train[3]

word_index = imdb.get_word_index()
word_index

reverse_word_index = dict()
for key, value in word_index.items():
  reverse_word_index[value] = key


for i in range(1,21):
  print (i,'->', reverse_word_index[i])

index = 3
message = ''
for code in x_train[index]:
    word = reverse_word_index.get(code - 3, '?')
    message += word + ' '
message

y_train[index]
maxlen = 200
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')
y_train[3]
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(maxlen,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train,
                    y_train,
                    epochs=1000,
                    batch_size=128,
                    validation_split=0.1)
plt.plot(history.history['acc'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_acc'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
scores = model.evaluate(x_test, y_test, verbose=1)
print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))