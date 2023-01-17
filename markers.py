import numpy as np

from imports import *


def change_names():
    for j in range (2,23):
        Path = 'C:/BazyDanych/MarkersDataset/jpg2png/0 (' + str(j) + ').png'
        background = cv2.imread(Path)
        background = cv2.resize(background, IMAGE_SIZE_CV)
        writePath = 'background_' + str(j+12) + '.png'
        cv2.imwrite(writePath, background)


def change_sizes():
    for j in range (0,33):
        Path = 'C:/BazyDanych/MarkersDataset/ArucoBackground/background_' + str(j) + '.png'
        background = cv2.imread(Path)
        background = cv2.resize(background, IMAGE_SIZE_CV)
        writePath = 'C:/BazyDanych/MarkersDataset/ArucoBackground/background_' + str(j) + '.png'
        cv2.imwrite(writePath, background)


def to_black_white(threshold,start,stop):
    for j in range (start,stop):
        Path = 'C:/BazyDanych/MarkersDataset/images/nav_' + str(j) + '.png'
        img = cv2.imread(Path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_black = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        writePath = 'C:/BazyDanych/MarkersDataset/images/black/nav_' + str(j) + '.png'
        cv2.imwrite(writePath, img_black)


def generate_markers():
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    #img = cv2.aruco.drawMarker(dictionary,1,100)
    for i in range (0,20):
        img = cv2.aruco.drawMarker(dictionary, i, 100)
        filename = 'aruco_4x4_50_' + str(i) + '.png'
        cv2.imwrite(filename, img)


def syntetic_images(Markersize,rangeStart,rangeStop):

    marker_shape = (Markersize,Markersize)

    for j in range(rangeStart,rangeStop):

        #random1 = np.random.randint(0, 5)
        backgroundrPath = 'C:/BazyDanych/MarkersDataset/ArucoBackground/background_' + str(10) + '.png'
        background = cv2.imread(backgroundrPath)
        background = cv2.resize(background, IMAGE_SIZE_CV)

        output = background.copy()

        while 1:
            positions = getCoordindates(output, marker_shape)
            if checkPositions(positions):
                break

        for k in range(0,5):
            random2 = np.random.randint(0, 33)
            markerPath = 'C:/BazyDanych/MarkersDataset/ArucoMarkers/aruco_4x4_50_' + str(random2) + '.png'
            marker = cv2.imread(markerPath)

            marker = cv2.resize(marker, marker_shape)
            output[positions[k,1]:positions[k,1] + positions[k,2], positions[k,0]:positions[k,0] + positions[k,3]] = marker

        name = 'nav_' + str(j)
        generateAnnotation(name,positions)
        writePath = 'C:/BazyDanych/MarkersDataset/images/' + name + '.png'
        cv2.imwrite(writePath, output)


def checkPositions(positions):
    isgood = True
    for i in range(0,5):
        for j in range(0,5):
            if i != j:
                if np.absolute(positions[i,0] - positions[j,0]) < positions[i,2] and np.absolute(positions[i,1] - positions[j,1]) < positions[i,3]:
                    isgood = False
    return isgood


def getCoordindates(output,marker_shape):
    pos = np.zeros(shape=(5, 4), dtype=int)
    for i in range(0,5):
        height1, width1, _ = output.shape
        height2, width2 = marker_shape
        pos[i, 0] = np.random.randint(0, width1 - width2)
        pos[i, 1] = np.random.randint(0, height1 - height2)
        pos[i, 2] = marker_shape[0]
        pos[i, 3] = marker_shape[1]
    return pos


def readJson(subDir, name):
    with open(os.path.join(homeDataDirectory,subDir, '{:s}.json'.format(name)), 'r') as fid:
        anno = json.load(fid)
    return anno


def readTXT(num):
    with open("C:/BazyDanych/MarkersDataset/annotations/labels/nav_" + str(num)+".txt",'r') as file:
        data = file.readlines()
    return data


def generateAnnotation(name,positions):
    template = readJson('ArucoAnnotation', 'template')

    output = template.copy()
    output['name'] = name
    output['width'] = 192
    output['height'] = 180
    #transform upper left corner, to middle of the bbox
    for i in range(0,5):
        positions[i, 0] = positions[i, 0] + positions[i, 2] / 2
        positions[i, 1] = positions[i, 1] + positions[i, 3] / 2

    for j in range(0, 5):
        output['objects'][j]['bbox']['x'] = float(positions[j,0]) /  output['width']
        output['objects'][j]['bbox']['y'] = float(positions[j,1]) /  output['height']
        output['objects'][j]['bbox']['width'] = float(positions[j,2]) / output['width']
        output['objects'][j]['bbox']['height'] = float(positions[j,3]) /  output['height']
    # Serializing json
    json_object = json.dumps(output, indent=4)
    with open("C:/BazyDanych/MarkersDataset/annotations/" + name + ".json", "w") as outfile:
        outfile.write(json_object)


def translateData_TXT_JSON():
    template = readJson('ArucoAnnotation', 'template')

    for i in range(0,14):
        output = template.copy()
        output['name'] = 'nav_' + str(i)
        output['width'] = 500
        output['height'] = 400
        data = readTXT(i)
        for j in range(0,5):
            parsedData =data[j].split(" ")
            output['objects'][j]['bbox']['x'] = float(parsedData[1])
            output['objects'][j]['bbox']['y'] = float(parsedData[2])
            output['objects'][j]['bbox']['width'] = float(parsedData[3])
            output['objects'][j]['bbox']['height'] = float(parsedData[4])

        # Serializing json
        json_object = json.dumps(output, indent=4)
        with open("C:/BazyDanych/MarkersDataset/annotations/nav_"+str(i)+".json", "w") as outfile:
            outfile.write(json_object)


def show_image (imageNum):
    path = "C:/BazyDanych/MarkersDataset/images/nav_" + str(imageNum) + ".png"
    img = cv2.imread(path)
    annotation = readJson('annotations','nav_'+str(imageNum))
    height, width, _ = img.shape

    for j in range(0,5):
        width = annotation['width']
        height = annotation['height']
        x = annotation['objects'][j]['bbox']['x']
        y = annotation['objects'][j]['bbox']['y']
        Xwidth = annotation['objects'][j]['bbox']['width']
        Yheight = annotation['objects'][j]['bbox']['height']

        leftUpX =  int((x - Xwidth/2) * width)
        leftUpY =  int((y - Yheight/2) * height)
        rightDownX = int((x + Xwidth/2) * width)
        rightDownY = int((y + Yheight/2) * height)
        leftUp = (leftUpX, leftUpY)
        rightDown = (rightDownX, rightDownY)
        img = cv2.rectangle(img, leftUp, rightDown, (0, 255, 0), thickness=2)
        cv2.namedWindow("picture", cv2.WINDOW_NORMAL)
    cv2.imshow('picture', img)
    cv2.waitKey(0)



