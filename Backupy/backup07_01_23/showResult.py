from imports import *
from model import loadModel
from dataProcessing import *


def showResult(model,num):

    image = loadImage(num)
    image = cv2.resize(image, IMAGE_SIZE_CV)
    image = image / 255.0
    result = model.predict(np.array([image]))
    output = image.copy()

    height, width, _ = output.shape

    for i in range(0, NUM_OF_WINDOWS):
        confidence = result[0, i * NUM_OF_WINDOW_PARAM + 0]
        centreX = result[0, i * NUM_OF_WINDOW_PARAM + 1] * width
        centreY = result[0, i * NUM_OF_WINDOW_PARAM + 2] * height
        boxWidth = result[0, i * NUM_OF_WINDOW_PARAM + 3] * width
        boxHeight = result[0, i * NUM_OF_WINDOW_PARAM + 4] * height

        leftUp = (int(centreX - boxWidth/2), int(centreY - boxHeight/2))
        rightDown = (int(centreX + boxWidth/2), int(centreY + boxHeight/2))

        #leftUp = (int( result[0, i * NUM_OF_WINDOW_PARAM + 1] * width), int(result[0, i * NUM_OF_WINDOW_PARAM + 2] * height))
        #rightDown = (int( result[0, i * NUM_OF_WINDOW_PARAM + 3] * width), int(result[0, i * NUM_OF_WINDOW_PARAM + 4] * height))

        output = cv2.rectangle(output, leftUp, rightDown, (0, 255, 0), thickness=2)
        output = cv2.putText(output, str(confidence), leftUp,cv2.FONT_HERSHEY_SIMPLEX,0.3, (0, 255, 0), thickness=1)

    cv2.namedWindow("picture", cv2.WINDOW_NORMAL)
    cv2.imshow('picture', output)
    cv2.waitKey(0)


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


#def showPrediction(img, prediction):
#    height, width, _ = img.shape
#
#    for j in range(0, 5):
#        conf = prediction[j * NUM_OF_WINDOW_PARAM + 0]
#        x = prediction[j * NUM_OF_WINDOW_PARAM + 1]
#        y = prediction[j * NUM_OF_WINDOW_PARAM + 2]
#        Xwidth = prediction[j * NUM_OF_WINDOW_PARAM + 3]
#        Yheight = prediction[j * NUM_OF_WINDOW_PARAM + 4]
#
#        leftUpX = int((x - Xwidth / 2) * width)
#        leftUpY = int((y - Yheight / 2) * height)
#        rightDownX = int((x + Xwidth / 2) * width)
#        rightDownY = int((y + Yheight / 2) * height)
#        leftUp = (leftUpX, leftUpY)
#        rightDown = (rightDownX, rightDownY)
#        img = cv2.rectangle(img, leftUp, rightDown, (0, 255, 0), thickness=2)
#        cv2.namedWindow("picture", cv2.WINDOW_NORMAL)
#    cv2.imshow('picture', img)
#    cv2.waitKey(0)

