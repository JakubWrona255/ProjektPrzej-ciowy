from imports import *


def standardConv2D_BN_LeakyRelu(dataIn,kernel=(3, 3), filters=32,momentum=0.9,stride=(1,1)):
    x = tf.keras.layers.Conv2D(activation='linear', kernel_size=kernel, filters=filters, padding="same",strides=stride)(dataIn)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=momentum)(x)
    return tf.keras.layers.LeakyReLU()(x)


def residualLayer(dataIn, kernel1=(1,1), kernel2=(3, 3), filters=32, momentum=0.9):
    skip = dataIn
    x = standardConv2D_BN_LeakyRelu(dataIn, kernel=kernel1, filters=int(filters/2), momentum=momentum)
    x = standardConv2D_BN_LeakyRelu(x, kernel=kernel2, filters=filters, momentum=momentum)
    return skip + x


def loadModel(path):
    return tf.keras.models.load_model(path)


def buildModelClassification():

    # input layer with corresponding dropout
    inputData = tf.keras.Input(shape=IMAGE_SIZE_TF)
    x = tf.keras.layers.SpatialDropout2D(rate=0.1)(inputData)

    #conv1
    x = standardConv2D_BN_LeakyRelu(x,kernel=(3,3),filters=128,momentum=0.9)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    #conv2
    x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=256, momentum=0.9)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    #conv3
    x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=256, momentum=0.9)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # conv4
    x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=256, momentum=0.9)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # conv5
    x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=256, momentum=0.9)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(360, activation='linear')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.05)(x)

    x = tf.keras.layers.Dense(256, activation='linear')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.05)(x)

    output = tf.keras.layers.Dense(2,activation=tf.keras.activations.softmax)(x)

    network = tf.keras.Model(inputs=inputData,outputs=output)

    #network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
                    loss=tf.keras.losses.huber,
                    metrics=tf.keras.metrics.binary_accuracy,
                    #run_eagerly=True
                    )
    network.summary()
    return network


def showModel(net,filename):
    color_map = defaultdict(dict)   #customizethecolours
    #color_map[tf.keras.layers.Input]['fill'] = '#581845'

    color_map[tf.keras.layers.SpatialDropout2D]['fill'] = '#FA4A25'
    color_map[tf.keras.layers.Dropout]['fill'] = '#FA4A25'

    #color_map[tf.keras.layers.Conv2D]['fill'] = '#003F5C'
    #color_map[tf.keras.layers.BatchNormalization]['fill'] = '#665191'
#    color_map[tf.keras.layers.LeakyReLU]['fill'] = '#D45087'
#
#    color_map[tf.keras.layers.MaxPooling2D]['fill'] = '#FF7C43'
#
#
    #color_map[tf.keras.layers.Dropout]['fill'] = '#03045e'
    #color_map[tf.keras.layers.Dense]['fill'] = '#fb5607'
    color_map[tf.keras.layers.Flatten]['fill'] = '#8D00DD'
    vk.layered_view(net,to_file=filename, legend=True, color_map=color_map)

