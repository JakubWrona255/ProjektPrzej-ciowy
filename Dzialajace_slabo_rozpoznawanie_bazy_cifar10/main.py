import tensorflow as tf
from tensorflow.python import keras
import idx2numpy
import time
import numpy as np
import matplotlib.pyplot as plt


(trainExamples, trainLabels), (testExamples, testLabels) = tf.keras.datasets.cifar10.load_data()

trainExamples = trainExamples / 255.0
testExamples = testExamples / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainExamples[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[trainLabels[i][0]])
plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(400,activation='relu'))
model.add(tf.keras.layers.Dense(10))

model.summary()

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

history = model.fit(trainExamples,trainLabels, epochs=10,
                    validation_data=(testExamples,testLabels))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(testExamples,  testLabels, verbose=2)

print(test_acc)

model.save('Model_CIFAR10_trained_10epochs')

#train_dataset = tf.data.Dataset.from_tensor_slices((trainExamples, trainLabels))
#test_dataset = tf.data.Dataset.from_tensor_slices((testExamples, testLabels))
#
#imageSize = (28,28)
#batchSize = 32
#shuffleBufferSize = 100
#
#train_dataset = train_dataset.shuffle(shuffleBufferSize).batch(batchSize)
#test_dataset = test_dataset.batch(batchSize)
#
#model = tf.keras.Sequential([
#    tf.keras.layers.Flatten(input_shape=imageSize),
#    tf.keras.layers.Dense(800,activation='relu'),
#    tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax)
#])
#
#
#model.compile(optimizer=tf.keras.optimizers.RMSprop(),
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['sparse_categorical_accuracy'])
#
#model.fit(train_dataset, epochs=10)
#model.evaluate(test_dataset)
#
#model.save('Model1')
#
#
##for i in range (0,20):
##    plt.imshow(arr[i])
##    plt.show()
##    time.sleep(1)




