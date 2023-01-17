import numpy as np

from imports import *
from model import buildModelClassification, buildModelLocalisation, loadModel
from dataProcessing import *
from showResult import showResult, showResultClassification
from markers import syntetic_images,change_sizes
from classification import *


def TrainModel(model):

    for i in range(0,10):
        dataTrain = datasetCreation(start=0,stop=500)
        history = model.fit(dataTrain,batch_size=MINI_BATCH_SIZE,epochs=EPOCHS_PER_BATCH)
        dataTrain = datasetCreation(start=501, stop=1000)
        history = model.fit(dataTrain, batch_size=MINI_BATCH_SIZE, epochs=EPOCHS_PER_BATCH)
        dataTrain = datasetCreation(start=1001, stop=1500)
        history = model.fit(dataTrain, batch_size=MINI_BATCH_SIZE, epochs=EPOCHS_PER_BATCH)
        dataTrain = datasetCreation(start=1501, stop=2000)
        history = model.fit(dataTrain, batch_size=MINI_BATCH_SIZE, epochs=EPOCHS_PER_BATCH)
        dataTrain = datasetCreation(start=2001, stop=2500)
        history = model.fit(dataTrain, batch_size=MINI_BATCH_SIZE, epochs=EPOCHS_PER_BATCH)
        dataTrain = datasetCreation(start=2501, stop=3000)
        history = model.fit(dataTrain, batch_size=MINI_BATCH_SIZE, epochs=EPOCHS_PER_BATCH)


        #plotHistory(history)

        model.save('Markers_5_last_one')

    model.save('Markers_5_0')


def tryModel(name):
    mod = loadModel(name)
    for i in range(1500,2000):
        showResultClassification(mod,i)


def plotHistory(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['mean_squared_error'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


def getHeatMap(picture,net):
    #size of 1080p - 1920 x 1080
    #size of network 384 x 360 (10 x 6 steps with 50% overlap on both axis)

    candidateMarkers = []

    height, width, _ = picture.shape
    numOfStepsVertical = 6
    numOfStepsHorizontal = 10
    stepLenVertical = int(height / numOfStepsVertical)
    stepLenHorizontal = int(width / numOfStepsHorizontal)

    #heatMap = np.zeros(shape=(numOfStepsVertical-1,numOfStepsHorizontal-1))

    for i in range(0,numOfStepsVertical-1):
        for j in range(0, numOfStepsHorizontal-1):

            heightSplitStart = i * stepLenVertical
            heightSplitStop = heightSplitStart + 2 * stepLenVertical
            widthSplitStart = j * stepLenHorizontal
            widthSplitStop = widthSplitStart + 2 * stepLenHorizontal

            subPicture = picture[heightSplitStart:heightSplitStop, widthSplitStart:widthSplitStop]

            picCopy = subPicture.copy()
            output = subPicture.copy()

            picCopy = picCopy / 255
            result = net.predict(np.array([picCopy]))

            res = np.argmax(result)
            if res == 0:
                text = "object"
            elif res == 1:
                text = "background"
            else:
                text = "error"

            #heatMap[i,j] = int(np.absolute(res-1))

            if res == 0:
                getMarkerPosition(subPicture,candidateMarkers)

            #output = cv2.putText(output, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), thickness=1)
            #cv2.namedWindow("picture", cv2.WINDOW_NORMAL)
            #cv2.imshow('picture', output)
            #cv2.waitKey(0)
            #print(heatMap)
    print(candidateMarkers)


def slidingWindow():
    img = cv2.imread('photo1.png')
    network = loadModel('Markers_5_last_one')
    getHeatMap(img,network)


def getMarkerPosition(subPicture,candidMarkers):

    absoluteX = 0
    absoluteY = 0

    #subPicture =  cv2.flip(subPicture,1)

    gray = cv2.cvtColor(subPicture, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = cv2.aruco.drawDetectedMarkers(subPicture.copy(), corners, ids)

    #print(corners,ids,rejectedImgPoints)

    plt.figure()
    plt.imshow(frame_markers)

    #print(len(ids))

    try:
        numOfIDS = len(ids)

    except:
        numOfIDS = 0
        print('error')

    for i in range(numOfIDS):
        c = corners[i][0]
        plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))

        candidMarkers.append([ids[i,0],c[:, 0].mean(),c[:, 1].mean()])

    plt.legend()
    #plt.show()

    return absoluteX,absoluteY


def readPosition(markerID):
    posData = {
        0: (233,235),
        1: (888,888),
        2: (233, 235),
        3: (233, 235),
        4: (233, 235),
        5: (233, 235),
        6: (233, 235),
        7: (233, 235),
        8: (233, 235),
        9: (233, 235),
        10: (233, 235),
        11: (233, 235),
        12: (233, 235),
    }
    return posData[markerID]


if __name__ == '__main__':
    pass
    #img = cv2.imread('photo2.png')
    #getMarkerPosition(img)

    slidingWindow()
    #syntetic_images_classification(100, 0, 500)
    #syntetic_images_classification(175, 501, 1000)
    #syntetic_images_classification(225, 1001, 1500)
    #syntetic_images_classification(250, 2001, 2500)
    #syntetic_images_classification(200, 2501, 3000)

    #net = buildModelClassification()
    #TrainModel(net)
    #tryModel('Markers_5_last_one')
