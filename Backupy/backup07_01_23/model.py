from imports import *


def standardConv2D_BN_LeakyRelu(dataIn,kernel=(3, 3), filters=32,momentum=0.9,stride=(1,1)):
    x = tf.keras.layers.Conv2D(activation='linear', kernel_size=kernel, filters=filters, padding="same",strides=stride)(dataIn)
    #x = tf.keras.layers.BatchNormalization(axis=1, momentum=momentum)(x)
    return tf.keras.layers.LeakyReLU()(x)


def residualLayer(dataIn, kernel1=(1,1), kernel2=(3, 3), filters=32, momentum=0.9):
    skip = dataIn
    x = standardConv2D_BN_LeakyRelu(dataIn, kernel=kernel1, filters=int(filters/2), momentum=momentum)
    x = standardConv2D_BN_LeakyRelu(x, kernel=kernel2, filters=filters, momentum=momentum)
    return skip + x


def loadModel(path):
    return tf.keras.models.load_model(path)


def buildModelLocalisation():

    # input layer with corresponding dropout
    inputData = tf.keras.Input(shape=IMAGE_SIZE_TF)
    x = tf.keras.layers.Dropout(rate=0.0)(inputData)

    #conv1
    x = standardConv2D_BN_LeakyRelu(x,kernel=(3,3),filters=32,momentum=0.9)

    #conv2
    x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=64, momentum=0.9,stride=(2,2))

    #residual1
    #x = residualLayer(dataIn=x,kernel1=(1,1),kernel2=(3,3),filters=64)

    #conv3
    x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=128, momentum=0.9, stride=(2, 2))

    # residual2
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=128)
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=128)

    # conv4
    x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=256, momentum=0.9, stride=(2, 2))

    # residual3
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=256)
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=256)
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=256)
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=256)

    ## conv5
    #x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=512, momentum=0.9, stride=(2, 2))
#
    ## residual4
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=512)
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=512)
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=512)
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=512)
#
    ## conv6
    #x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=1024, momentum=0.9, stride=(2, 2))
#
    ## residual5
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=1024)
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=1024)
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=1024)
    #x = residualLayer(dataIn=x, kernel1=(1, 1), kernel2=(3, 3), filters=1024)
#
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(1024, activation='linear')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    #x = tf.keras.layers.Dropout(rate=0.05)(x)

    x = tf.keras.layers.Dense(1024, activation='linear')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    #x = tf.keras.layers.Dropout(rate=0.05)(x)

    #output = tf.keras.layers.Dense(OUTPUT_VECTOR_SIZE,activation=tf.keras.activations.sigmoid)(x)
    output = tf.keras.layers.Dense(OUTPUT_VECTOR_SIZE,activation=tf.keras.activations.linear)(x)

    network = tf.keras.Model(inputs=inputData,outputs=output)

    network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
                    loss=tf.keras.losses.huber,
                    metrics=tf.keras.metrics.mean_squared_error,
                    run_eagerly=True
                    )
    network.summary()
    return network


def buildModelClassification():

    # input layer with corresponding dropout
    inputData = tf.keras.Input(shape=IMAGE_SIZE_TF)
    x = tf.keras.layers.Dropout(rate=0.1)(inputData)

    #conv1
    x = standardConv2D_BN_LeakyRelu(x,kernel=(3,3),filters=128,momentum=0.9)

    #conv2
    x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=256, momentum=0.9,stride=(2,2))

    #conv3
    x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=256, momentum=0.9, stride=(2, 2))

    # conv4
    x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=256, momentum=0.9, stride=(2, 2))

    # conv5
    x = standardConv2D_BN_LeakyRelu(x, kernel=(3, 3), filters=256, momentum=0.9, stride=(2, 2))

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(512, activation='linear')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.05)(x)

    x = tf.keras.layers.Dense(256, activation='linear')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.05)(x)

    output = tf.keras.layers.Dense(2,activation=tf.keras.activations.softmax)(x)

    network = tf.keras.Model(inputs=inputData,outputs=output)

    network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
                    loss=tf.keras.losses.huber,
                    metrics=tf.keras.metrics.mean_squared_error,
                    run_eagerly=True
                    )
    network.summary()
    return network

