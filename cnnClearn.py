from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
import matplotlib.pyplot as plt
from keras import backend as K
import random
from sklearn.decomposition import PCA


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

accList = []
trainSizeList = []
lossList = []
rbfSizeList = []
runType = "complex"
fileName = "WifiData7"

# y = np.loadtxt(fileName + "yValues.txt")
y = np.loadtxt("WifiData7yValues.txt")


sampleSize = 400

if (runType == "absolute"):
    x = np.loadtxt(str(sampleSize) + "AbsoluteSTFT.txt")
elif (runType == "complex"):
    x1 = np.loadtxt(fileName + str(sampleSize) + "RealFFT.txt")
    x2 = np.loadtxt(fileName + str(sampleSize) + "ComplexFFT.txt")
    sampleSize *= 2
    xSize = sampleSize
    x = np.ones((np.shape(x1)[0], xSize))
    x[:, 0:xSize:2] = x1
    x[:, 1:xSize:2] = x2
    #x[:,0:math.floor(xSize/2)] = x1
    #x[:,math.ceil(xSize/2):xSize] = x2
elif (runType == "magPhase"):
    x1 = np.loadtxt(fileName + str(sampleSize) + "Magnitude.txt")
    x2 = np.loadtxt(fileName + str(sampleSize) + "Phase.txt")
    sampleSize *= 2
    xSize = sampleSize
    x = np.ones((np.shape(x1)[0], xSize))
    x[:, 0:xSize:2] = x1
    x[:, 1:xSize:2] = x2
    # x[:,0:math.floor(xSize/2)] = x1
    # x[:,math.ceil(xSize/2):xSize] = x2
elif (runType == "original"):
    x = np.loadtxt(str(sampleSize) + "Original.txt")
else:
    x = np.loadtxt(str(sampleSize) + "Base.txt")

#begin toy code

# toyX = np.loadtxt('toyX',  delimiter=',')
# toyY = np.loadtxt('toyY', delimiter=',')
# x = np.loadtxt('toyX',  delimiter=',')
# y = np.loadtxt('toyY', delimiter=',')
# sampleSize = 800
#End toy code

maxSize = np.size(y)
y = y.astype(int)


x2 = x[0:maxSize,0:sampleSize]


ignoringClasses = np.where(y >= 8)
restOfClassLocation = [i for i in range(maxSize) if i not in ignoringClasses[0]]
y = y[restOfClassLocation]
x2 = x2[restOfClassLocation]

maxSize = np.size(y)

randClassLocation = np.where(y >= 4)
restOfClassLocation = [i for i in range(maxSize) if i not in randClassLocation[0]]
randClassY = y[randClassLocation]
randClassX = x2[randClassLocation]
y = y[restOfClassLocation]
x2 = x2[restOfClassLocation]

# size = 1190
maxSize = np.size(y)  # 1178

percentTrain = 0.5
trainSize = round(percentTrain * maxSize)
avgAccuracy = 0


# x2 = principalComponents
X,Y = unison_shuffled_copies(x2,y)
# X = x2
# Y = y

num_classes = max(y) + 1
print(num_classes)

xTrain = X[0:trainSize]
xTest = X[trainSize: maxSize]
yTrainTemp = Y[0:trainSize]
yTestTemp = Y[trainSize:maxSize]

xTrain = np.expand_dims(xTrain, axis=2)
xTest = np.expand_dims(xTest, axis=2)
randClassX = np.expand_dims(randClassX, axis=2)

yTrain = to_categorical(yTrainTemp, num_classes)
yTest = to_categorical(yTestTemp, num_classes)

input_shape = (sampleSize, 1)

print()
print("shape xTest: ", np.shape(xTest))
print("shape xTr: ", np.shape(xTrain))
print("shape yTest: ", np.shape(yTest))
print("shape yTr: ", np.shape(yTrain))
print('input shape:', input_shape)

model = Sequential()
#Cnn
model.add(Conv1D(16, kernel_size=(5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv1D(64, (1), activation='relu'))
model.add(MaxPooling1D(pool_size=(1)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(xTrain, yTrain,
          batch_size=50,
          epochs=80,
          verbose=1,
          validation_data=(xTest, yTest))

#Last Layer Output Here
xTrainPredict = model.predict(xTrain)
xTestPredict = model.predict(xTest)
xUnknownClassPredict = model.predict(randClassX)
#2nd Last layer output here
# get2ndLastLayer = K.function([model.layers[0].input],
#                                   [model.layers[-2].output])
# xTrainPredict = get2ndLastLayer([xTrain])[0]
# xTestPredict = get2ndLastLayer([xTest])[0]
# xUnknownClassPredict = get2ndLastLayer([randClassX])[0]
# print("Shape of 2nd last layer: ", np.shape(xTestPredict))
#End 2ndLayer

xTrainAvg = [[] for arr in range(num_classes)]
xTrainVar = [[] for arr in range(num_classes)]

for i in range(num_classes):
    classLocation = np.argwhere(yTrainTemp == i)
    xTrainAvg[i]= np.average(xTrainPredict[classLocation,:] ,0)[0]
    # temp1  = xTrainPredict[classLocation]
    # temp2 = np.cov(xTrainPredict[classLocation])
    # temp3 = np.cov(xTrainPredict[classLocation])[0]
    # xTrainVar[i] = np.cov(xTrainPredict[classLocation[0]])[0]
#     print("------**********************___++++++")
# print(xTrainAvg)
# print(xTrainVar)
#
# print(xTrainAvg[1])
# print(xTrainVar[1])

score = model.evaluate(xTest, yTest)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Save results. We need:
#1. Prediction average and variance.
np.savetxt('zeroShotDataSet/xTrainAvg', xTrainAvg, delimiter=',')
# np.savetxt('zeroShotDataSet/xTrainVar', xTrainVar, delimiter=',')

#2. Prediction of all test sets. Plus which set it's supposed to be!
np.savetxt('zeroShotDataSet/xTestPredict', xTestPredict, delimiter=',')
np.savetxt('zeroShotDataSet/yTest', yTest, delimiter=',')

#3. Prediction of unused set (random). (Include which set it's supposed to be when > 1 class)
np.savetxt('zeroShotDataSet/xUnknownClassPredict', xUnknownClassPredict, delimiter=',')
np.savetxt('zeroShotDataSet/randClassY', randClassY, delimiter=',')


#Maybe randClass to tell us just which class was picked.
#TODO Logan, make all classes have equal N Training/Testing Samples.