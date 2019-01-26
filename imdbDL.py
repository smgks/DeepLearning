import numpy as np
from keras.datasets import imdb
from keras import optimizers
from keras import losses
from keras import metrics
from keras import models
from keras import layers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

def dedcode_messege_byindex(index):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[index]])
    print(decoded_review)


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))



x_val = x_train[:10000]
partical_x_train = x_train[10000:]

y_val = y_train[:10000]
partical_y_train = y_train[10000:]

model.compile(optimizer=optimizers.rmsprop(lr=0.01),
              loss=losses.binary_crossentropy,
              metrics=['acc'])
history = model.fit(partical_x_train,
                    partical_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_val,y_val))
