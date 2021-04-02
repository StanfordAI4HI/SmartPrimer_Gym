import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

print("loading in the data")
data = np.genfromtxt('/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/gym_SmartPrimer/examples/observations6.csv', delimiter=',')

print("loaded in the data")
labels = np.genfromtxt('/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/gym_SmartPrimer/examples/labels6.csv', delimiter=',')

labels = pd.Series(labels)
countrs = labels.value_counts()
print(countrs)


train = data[0:70000, :]
train_labels = labels[0:70000]

test = data[70000:,:]
test_labels = labels[70000:]

print('starting to train the model')
model = keras.Sequential(([
    keras.layers.Dense(16, activation=tf.nn.tanh),
    keras.layers.Dense(16, activation=tf.nn.tanh),
		keras.layers.Dense(4, activation=tf.nn.softmax)]))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test,  test_labels, verbose=2)

print('Test accuracy:', test_acc)

model.save('behave_cloning6.h5')
