from imports import *
from model import loadModel


def scanPicture(picture,net):
    #size of 1080p - 1920 x 1080
    #size of network 384 x 360 (10 x 6 steps with 50% overlap on both axis)

    candidateMarkers = []

    height, width, _ = picture.shape
    numOfStepsVertical = 6
    numOfStepsHorizontal = 10

    stepLenVertical = int(height / numOfStepsVertical)
    stepLenHorizontal = int(width / numOfStepsHorizontal)

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
            picCopy =cv2.resize(picCopy, IMAGE_SIZE_CV)
            result = net.predict(np.array([picCopy]),verbose=0)

            res = np.argmax(result)
            if res == 0:
                text = "object"
            elif res == 1:
                text = "background"
            else:
                text = "error"

            if res == 0:
                subPicPos = (heightSplitStart,widthSplitStart)
                getMarkerPosition(subPicture,candidateMarkers,subPicPos)

            #output = cv2.putText(output, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), thickness=1)
            #cv2.namedWindow("picture", cv2.WINDOW_NORMAL)
            #cv2.imshow('picture', output)
            #cv2.waitKey(0)

    #print(candidateMarkers)
    return candidateMarkers


def getMarkerPosition(subPicture,candidMarkers,subPicturePosition):
    frameAbsY, frameAbsX = subPicturePosition

    gray = cv2.cvtColor(subPicture, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    frame_markers = cv2.aruco.drawDetectedMarkers(subPicture.copy(), corners, ids)
    #plt.figure()
    #plt.imshow(frame_markers)

    try:
        numOfIDS = len(ids)
    except:
        numOfIDS = 0
        #print('No markers found on proposed image')

    for i in range(numOfIDS):
        c = corners[i][0]
        #plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))

        absX = frameAbsX + c[:, 0].mean()
        absY = frameAbsY + c[:, 1].mean()
        candidMarkers.append([ids[i,0],absX,absY])

    #plt.legend()
    #plt.show()


def fuseCandidatePos(candidatePos):
    checkedPos = []
    numOfCandidates = len(candidatePos)

    for i in range(0,USED_MARKERS):

        checkedID = i
        idDetected = False
        tempX = []
        tempY = []

        for j in range(0,numOfCandidates):
            if candidatePos[j][0] == checkedID:
                idDetected = True
                tempX.append(candidatePos[j][1])
                tempY.append(candidatePos[j][2])

        if idDetected:
            averX = sum(tempX) / len(tempX)
            averY = sum(tempY) / len(tempY)
            checkedPos.append([checkedID,averX,averY])

    return checkedPos


def outlineMarkers(picture,positions):
    output = picture.copy()

    for i in range(0,len(positions)):
        text = 'id: ' + str(positions[i][0])
        textPos = (int(positions[i][1]+50), int(positions[i][2]+50))
        output = cv2.putText(output, text, textPos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    #cv2.namedWindow("picture",cv2.WINDOW_NORMAL)
    #cv2.imshow('picture', output)
    #cv2.resizeWindow("picture",1440,810)
    #cv2.waitKey(0)

    return output


def procesVideo(net):
    capture = cv2.VideoCapture('vid2.mp4')

    # Properties
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = capture.get(cv2.CAP_PROP_FPS)

    # Video Writer
    video_writer = cv2.VideoWriter('vid_processed_3.avi', cv2.VideoWriter_fourcc('P', 'I', 'M', '1'), fps, (width, height))

    candidatePositions = []
    candidatePositions_prev1 = []
    candidatePositions_prev2 = []
    combinedCandidates = []
    certainPositions = []

    for frame_idx in range(0,int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):

        ret, frame = capture.read()

        if frame_idx % 2 == 0:

            candidatePositions_prev2 = candidatePositions_prev1
            candidatePositions_prev1 = candidatePositions
            candidatePositions = scanPicture(frame, net)
            combinedCandidates = candidatePositions + candidatePositions_prev1 + candidatePositions_prev2
            certainPositions = fuseCandidatePos(combinedCandidates)

        frameProcessed = outlineMarkers(frame, certainPositions)
        frameProcessed = drawBarriers(frameProcessed,certainPositions)
        video_writer.write(frameProcessed)

        #cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        #cv2.imshow('frame', frameProcessed)
        #cv2.resizeWindow('frame',1440,810)
        #if cv2.waitKey(10) & 0xFF == ord('q'):
        #    break

        print(frame_idx)
        #if frame_idx > 200:
        #    break

    capture.release()
    video_writer.release()
    cv2.destroyAllWindows()


def drawBarriers(pic,positions):

    green = (0, 255, 0)
    red = (0, 0, 255)

    posData = {
        0: (0, 0),
        1: (0, 0),
        2: (0, 0),
        3: (0, 0),
        4: (0, 0),
        5: (0, 0),
        6: (0, 0),
        7: (0, 0),
        8: (0, 0),
        9: (0, 0),
        10:(0, 0),
        11:(0, 0),
        12:(0, 0),
    }

    for i in range(0,len(positions)):
        id = positions[i][0]
        posData[id] = (int(positions[i][1]),int(positions[i][2]))

    output = pic.copy()

    if posData[1][0] != 0 and posData[5][0] != 0:

        if intersect(posData[3][0], posData[4][0], posData[1][0], posData[5][0], posData[3][1], posData[4][1], posData[1][1], posData[5][1]) \
                or intersect(posData[0][0], posData[2][0], posData[1][0], posData[5][0], posData[0][1], posData[2][1], posData[1][1], posData[5][1]) \
                or intersect(posData[0][0], posData[3][0], posData[1][0], posData[5][0], posData[0][1], posData[3][1], posData[1][1], posData[5][1]) \
                or intersect(posData[2][0], posData[4][0], posData[1][0], posData[5][0], posData[2][1], posData[4][1], posData[1][1], posData[5][1]):
            colour34 = green
            colour02 = green
            colour03 = green
            colour24 = green
        else:
            colour34 = red
            colour02 = red
            colour03 = red
            colour24 = red
    else:
        colour34 = green
        colour02 = green
        colour03 = green
        colour24 = green

    #lines of polygon
    try:
        if posData[0][0] != 0 and posData[2][0] != 0:
            output = cv2.line(output, posData[0], posData[2], colour02, 2)

        if posData[0][0] != 0 and posData[3][0] != 0:
            output = cv2.line(output, posData[0], posData[3], colour03, 2)

        if posData[3][0] != 0 and posData[4][0] != 0:
            output = cv2.line(output, posData[3], posData[4], colour34, 2)

        if posData[2][0] != 0 and posData[4][0] != 0:
            output = cv2.line(output, posData[2], posData[4], colour24, 2)
    except:
        pass

    #distance of id5 from id1
    try:
        if posData[1][0] != 0 and posData[5][0] != 0:
            output = cv2.line(output, posData[1], posData[5], (128, 0, 128), 2)
            textPos = (int(float(posData[1][0] + posData[5][0]) / 2), int(float(posData[1][1] + posData[5][1]) / 2))
            text = str(int(np.sqrt((posData[1][0] - posData[5][0]) ** 2 + (posData[1][1] - posData[5][1]) ** 2)))
            output = cv2.putText(output, text, textPos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    except:
        pass

    return output


def ccw(Ax,Bx,Cx,Ay,By,Cy):
    return (Cy-Ay) * (Bx-Ax) > (By-Ay) * (Cx-Ax)


# Return true if line segments AB and CD intersect
def intersect(Ax,Bx,Cx,Dx,Ay,By,Cy,Dy):
    return ccw(Ax,Cx,Dx,Ay,Cy,Dy) != ccw(Bx,Cx,Dx,By,Cy,Dy) and ccw(Ax,Bx,Cx,Ay,By,Cy) != ccw(Ax,Bx,Dx,Ay,By,Dy)


def runSystem():

    network = loadModel('Markers_predetector')

    procesVideo(network)
    #img = cv2.imread('photo4.png')

    #candidatePositions = scanPicture(img,network)
    #certainPositions = fuseCandidatePos(candidatePositions)
    #print(certainPositions)
    #imgProcessed = outlineMarkers(img, certainPositions)
