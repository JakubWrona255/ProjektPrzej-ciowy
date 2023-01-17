import imports
from markers import *


def syntetic_images_classification(Markersize,rangeStart,rangeStop):

    marker_shape = (Markersize,Markersize)

    for j in range(rangeStart,rangeStop):

        random1 = np.random.randint(0, 33)
        backgroundrPath = 'C:/BazyDanych/MarkersDataset/ArucoBackground/background_' + str(random1) + '.png'
        background = cv2.imread(backgroundrPath)
        background = cv2.resize(background, IMAGE_SIZE_CV)

        output = background.copy()

        position = getCoordindatesClassification(output,marker_shape)

        decision = np.random.randint(0,2)

        if decision == 1:
            random2 = np.random.randint(0, 33)
            markerPath = 'C:/BazyDanych/MarkersDataset/ArucoMarkers/aruco_4x4_50_' + str(random2) + '.png'
            marker = cv2.imread(markerPath)
            marker = cv2.resize(marker, marker_shape)
            output[position[1]:position[1] + position[2], position[0]:position[0] + position[3]] = marker

        name = 'nav_' + str(j)
        generateAnnotationClassification(name,position,decision)
        writePath = 'C:/BazyDanych/MarkersDataset/images/classification/' + name + '.png'
        cv2.imwrite(writePath, output)


def generateAnnotationClassification(name,positions,decision):
    template = readJson('ArucoAnnotation', 'template_classification')

    output = template.copy()
    output['name'] = name
    output['width'] = 500
    output['height'] = 400
    if decision == 1:
        output['objects'] = 1
        output['background'] = 0
    else:
        output['objects'] = 0
        output['background'] = 1

    # Serializing json
    json_object = json.dumps(output, indent=4)
    with open("C:/BazyDanych/MarkersDataset/annotations/classification/" + name + ".json", "w") as outfile:
        outfile.write(json_object)


def getCoordindatesClassification(output,marker_shape):

    pos = np.zeros(shape=4, dtype=int)
    height1, width1, _ = output.shape
    height2, width2 = marker_shape
    pos[0] = np.random.randint(0, width1 - width2)
    pos[1] = np.random.randint(0, height1 - height2)
    pos[2] = marker_shape[0]
    pos[3] = marker_shape[1]
    return pos
