from imports import *
from dataProcessing import *
from classification import *

from model import buildModelClassification, loadModel, showModel
from showResult import showResultClassification
from markers import syntetic_images,change_sizes
from navigation import runSystem


def TrainModel(model):

    for i in range(0,10):
        dataTrain = datasetCreation(start=0,stop=100)
        history = model.fit(dataTrain,batch_size=MINI_BATCH_SIZE,epochs=EPOCHS_PER_BATCH)

        plotHistory(history)
        model.save('Markers_6_last_one')

    model.save('Markers_6_0')


def TrainModelExperimental(model):

    #for i in range(0,10):

    dataTrain = datasetCreation(start=0,stop=2900)
    dataVal = datasetCreation(start=2901, stop=3000)
    history = model.fit(dataTrain,batch_size=MINI_BATCH_SIZE,epochs=EPOCHS_PER_BATCH,validation_data=dataVal)

    #history = model.fit(dataTrain, batch_size=MINI_BATCH_SIZE, epochs=EPOCHS_PER_BATCH)
    print(history.history.keys())
    plotHistory(history)

    #model.save('Markers_6_last_one')

    #model.save('Markers_6_0')


def tryModel(name):
    mod = loadModel(name)
    for i in range(0,500):
        showResultClassification(mod,i)


def plotHistory(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


if __name__ == '__main__':
    pass

    #runSystem()

    #model = buildModelClassification()
    #showModel(model,"model_1.png")

    #syntetic_images_classification(40, 0, 500)
    #syntetic_images_classification(42, 500, 1000)
    #syntetic_images_classification(44, 1000, 1500)
    #syntetic_images_classification(46, 1500, 2000)
    #syntetic_images_classification(48, 2000, 2500)
    #syntetic_images_classification(50, 2500, 3000)

    net = buildModelClassification()
    TrainModelExperimental(net)

