from imports import *
from model import loadModel
from dataProcessing import *


def showResultClassification(model,num):

    image = loadImage(num)
    image = cv2.resize(image, IMAGE_SIZE_CV)
    image = image / 255.0
    result = model.predict(np.array([image]))
    output = image.copy()

    height, width, _ = output.shape

    res = np.argmax(result)
    if res == 0:
        text = "object"
    elif res == 1:
        text = "background"
    else:
        text = "error"

    output = cv2.putText(output, text, (10,10),cv2.FONT_HERSHEY_SIMPLEX,0.3, (0, 255, 0), thickness=1)

    cv2.namedWindow("picture", cv2.WINDOW_NORMAL)
    cv2.imshow('picture', output)
    cv2.waitKey(0)

