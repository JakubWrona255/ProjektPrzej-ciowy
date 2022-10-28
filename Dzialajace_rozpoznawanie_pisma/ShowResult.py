import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

model = tf.keras.models.load_model('Model1.')

image = cv.imread('pismo_4.png')
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#_,image = cv.threshold(image,120,1,cv.THRESH_BINARY_INV)
image = 255-image

imageR = np.reshape(image, (1, 28, 28))
prediction = np.argmax(model.predict(imageR))
print(prediction)

plt.imshow(image)
plt.title(prediction)
plt.show()
