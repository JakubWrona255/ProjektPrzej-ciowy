import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

model = tf.keras.models.load_model('Model_CIFAR10_trained_10epochs')

image = cv.imread('dog_1.png')

image = image

imageR = np.expand_dims(image,0)
prediction = np.argmax(model.predict(imageR))

print(prediction)

plt.imshow(image)
plt.title(prediction)
plt.show()
