from imports import *


def normalizeImg(image):
    return tf.cast(image,tf.float32)/255.0


def loadImage(num):
    path = imagesPath + "nav_" + str(num) + ".png"
    img = cv2.imread(path)
    return img


def loadImages(data_start, data_stop):
    images = []
    for i in range(data_start,data_stop):
        img = loadImage(i)
        img = cv2.resize(img, IMAGE_SIZE_CV)
        img = normalizeImg(img)
        images.append(img)
    return images


def loadAnnotation(num):
    name = 'nav_' + str(num)
    with open(os.path.join(annoPath, '{:s}.json'.format(name)), 'r') as fid:
        anno = json.load(fid)
    return anno


def loadAnnotationsClassification(data_start, data_stop):

    dataLen = data_stop - data_start
    anno = np.zeros(shape=(dataLen,2))

    for i in range(data_start,data_stop):
        annoRaw = loadAnnotation(i)

        anno[i-data_start, 0] = annoRaw['objects']
        anno[i-data_start, 1] = annoRaw['background']

    return anno


def datasetCreation(start,stop):
    images = loadImages(data_start=start, data_stop=stop)
    annotations = loadAnnotationsClassification(data_start=start, data_stop=stop)
    #print(images[0])
    #print(annotations[0])
    Dataset = tf.data.Dataset.from_tensor_slices((images, annotations)).shuffle(SHUFFLE_BUFFER_SIZE).batch(MINI_BATCH_SIZE)

    return Dataset


