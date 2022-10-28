import tensorflow as tf
from tensorflow.python import keras
import idx2numpy
import time
import numpy as np
import matplotlib.pyplot as plt


DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

path = tf.keras.utils.get_file('mnist.npz', DATA_URL)

with np.load(path) as data:
    trainExamples = data['x_train']
    trainLabels = data['y_train']
    testExamples = data['x_test']
    testLabels = data['y_test']

train_dataset = tf.data.Dataset.from_tensor_slices((trainExamples, trainLabels))
test_dataset = tf.data.Dataset.from_tensor_slices((testExamples, testLabels))

imageSize = (28,28)
batchSize = 32
shuffleBufferSize = 100

train_dataset = train_dataset.shuffle(shuffleBufferSize).batch(batchSize)
test_dataset = test_dataset.batch(batchSize)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=imageSize),
    tf.keras.layers.Dense(800,activation='relu'),
    tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax)
])


model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=10)
model.evaluate(test_dataset)

model.save('Model1')


#for i in range (0,20):
#    plt.imshow(arr[i])
#    plt.show()
#    time.sleep(1)




