from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import plot3d

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

startAt = 0
sampleSize = 400
startShape = sampleSize - startAt

#TODO: test code Logan
x = np.loadtxt('datasets/finalProcessed/normalizeXAbs.txt',  delimiter=',')
y = np.loadtxt('datasets/finalProcessed/processedY.txt',  delimiter=',')
y = y.astype(int)
sampleSize = 400
#TODO: test code Logan
ySize = np.size(y)
x2 = x[0:ySize, startAt:sampleSize]
#RandomClass Select
randClassLocation = np.where(y >= 6)
restOfClassLocation = [i for i in range(ySize) if i not in randClassLocation[0]]
randClassY = y[randClassLocation]
randClassX = x2[randClassLocation]
y = y[restOfClassLocation]
x2 = x2[restOfClassLocation]
#End randomClass Select
ySize = np.size(y)

X,Y = unison_shuffled_copies(x2,y)
percentTrain = .6
trainSize = round(percentTrain * ySize)
xTrain = X[0:trainSize]
xTest = X[trainSize: ySize]
yTrain = Y[0:trainSize]
yTest = Y[trainSize:ySize]
print(np.shape(xTrain))



deep = True
secondLayer = 200
thirdLayer = 50
fourthLayer = 10
middleLayer = 3
input_img = Input(shape=(startShape,))
regularizerValue = 10e-6
#TODO regularizer not used above
#------------------ No 0      No1     No2      No3          No4       No5      No6     No7          No8              No9          No10
activationNames = ["softmax", "elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "exponential", "linear"]
ind = 10
activationFunction = activationNames[10]
decodedActivation = activationNames[6]
outerActivation = activationNames[5]
innerActivation = activationNames[10]

if(deep):
    encoded = Dense(secondLayer, activation=activationFunction)(input_img)
    encoded = Dense(thirdLayer, activation=activationFunction)(encoded)
    encoded = Dense(fourthLayer, activation=activationFunction)(encoded)
    encoded = Dense(middleLayer, activation=activationFunction)(encoded)
    decoded = Dense(fourthLayer, activation=activationFunction)(encoded)
    decoded = Dense(thirdLayer, activation=activationFunction)(encoded)
    decoded = Dense(secondLayer, activation=activationFunction)(encoded)
    decoded = Dense(startShape, activation=decodedActivation)(decoded)
else:
    encoded = Dense(secondLayer, activation=activationFunction)(input_img)
    decoded = Dense(startShape, activation=decodedActivation)(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(startShape,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
# decoder = Model(encoded_input, decoder_layer(encoded_input))
# autoencoder.compile(optimizer='adadelta', loss='mean_squared_error') #0.008
# autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error') #0.006
#1.6

# autoencoder.compile(optimizer='adadelta', loss='mean_absolute_percentage_error') #0.36
# autoencoder.compile(optimizer='adadelta', loss='mean_squared_logarithmic_error') #0.0043
# autoencoder.compile(optimizer='adadelta', loss='squared_hinge') #0.59
# autoencoder.compile(optimizer='adadelta', loss='hinge') #0.73
# autoencoder.compile(optimizer='adadelta', loss='categorical_hinge') #0.0013 ****
# autoencoder.compile(optimizer='adadelta', loss='logcosh') #0.0041
#Different Optimizer, loss?
# autoencoder.compile(optimizer='sgd', loss='mean_absolute_error') #0.0029
# autoencoder.compile(optimizer='RMSprop', loss='mean_absolute_error') #0.0031
#3.1
autoencoder.compile(optimizer='Adagrad', loss='mean_absolute_error') #0.0027
#4.2
# autoencoder.compile(optimizer='Adam', loss='mean_absolute_error') #0.0028
# autoencoder.compile(optimizer='Adamax', loss='mean_absolute_error') #0.0015* 2.1848e-04
# autoencoder.compile(optimizer='Nadam', loss='mean_absolute_error') #0.0014
#3.28

autoencoder.fit(xTrain, xTrain,
                epochs=20,
                batch_size=10,
                shuffle=True,
                validation_data=(xTest, xTest))

# encode and decode some digits
encoded_imgs = encoder.predict(xTest)
encoded_imgs_train = encoder.predict(xTrain)
encoded_imgs2 = autoencoder.predict(xTest)
encoded_imgs_unknown = encoder.predict(randClassX)


#Encoder fit
#Average encoded_imgs for each class
numberOfClasses = max(yTrain)
numberOfSamplesTrain = np.size(yTrain)
classAverages = np.zeros((numberOfClasses,middleLayer))
yAverages = np.zeros((numberOfSamplesTrain,middleLayer))
for classIndex in range(numberOfClasses):
    classLocation = np.where(yTrain == classIndex)
    classAverages[classIndex] = np.average(encoded_imgs_train[classLocation], 0)
    yAverages[classLocation] =  classAverages[classIndex]
    print(classIndex, ": ", classAverages[classIndex]) #Verbose

distanceMatrix = np.zeros((numberOfClasses, numberOfClasses))
for classIndex in range(numberOfClasses):
    for classIndex2 in range(numberOfClasses):
        distanceMatrix[classIndex, classIndex2] = sum((classAverages[classIndex] - classAverages[classIndex2])**2)
print(distanceMatrix)
print("Average Distanc x100: ", np.average(distanceMatrix)*100)

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#Pushing away code
newAverages = np.zeros((numberOfClasses,middleLayer))
step = 0.0005
exponential = True
exponent = 2
for classIndex in range(numberOfClasses):
    for classIndex2 in range(numberOfClasses):
        if(classIndex == classIndex2):
            continue
        difference = classAverages[classIndex] - classAverages[classIndex2]
        if(exponential):
            for valueIndex in range(middleLayer):
                if (difference[valueIndex]<0):
                    difference[valueIndex] = (difference[valueIndex]**exponent)*-1
                else:
                    difference[valueIndex] = (difference[valueIndex]**exponent)
        newAverages[classIndex] += step/difference #linear
    newAverages[classIndex] += classAverages[classIndex]
#maybe normalize averages
#Reassign y averages
for classIndex in range(numberOfClasses):
    classLocation = np.where(yTrain == classIndex)
    yAverages[classLocation] =  newAverages[classIndex]
    print(classIndex, ": ", newAverages[classIndex]) #Verbose

distanceMatrix = np.zeros((numberOfClasses, numberOfClasses))
for classIndex in range(numberOfClasses):
    for classIndex2 in range(numberOfClasses):
        distanceMatrix[classIndex, classIndex2] = sum((newAverages[classIndex] - newAverages[classIndex2])**2)
print(distanceMatrix)
print("Average Distanc x100: ", np.average(distanceMatrix)*100)



encoder.compile(optimizer='Adagrad', loss='mean_absolute_error')

encoder.fit(xTrain, yAverages,
                epochs=20,
                batch_size=10,
                shuffle=True)

plot3d.plot3D(encoded_imgs, yTest,0)
# plot3d.plot3D(encoded_imgs_train, yTrain,1)
plot3d.plot3D(encoded_imgs_unknown, randClassY,2)


encoded_imgs = encoder.predict(xTest)
encoded_imgs_train = encoder.predict(xTrain)
encoded_imgs2 = autoencoder.predict(xTest)
encoded_imgs_unknown = encoder.predict(randClassX)


plot3d.plot3D(encoded_imgs, yTest,3)
# plot3d.plot3D(encoded_imgs_train, yTrain,4)
plot3d.plot3D(encoded_imgs_unknown, randClassY,5)


for classIndex in range(numberOfClasses):
    classLocation = np.where(yTrain == classIndex)
    classAverages[classIndex] = np.average(encoded_imgs_train[classLocation], 0)
    yAverages[classLocation] =  classAverages[classIndex]
    print(classIndex, ": ", classAverages[classIndex]) #Verbose


distanceMatrix = np.zeros((numberOfClasses, numberOfClasses))
for classIndex in range(numberOfClasses):
    for classIndex2 in range(numberOfClasses):
        distanceMatrix[classIndex, classIndex2] = sum((classAverages[classIndex] - classAverages[classIndex2])**2)
print(distanceMatrix)
print("Average Distanc x100: ", np.average(distanceMatrix)*100)

plt.show()



# np.savetxt('datasets/temp/encodedImgs', encoded_imgs, delimiter=',')
# np.savetxt('datasets/temp/encodedImgsTrain', encoded_imgs_train, delimiter=',')
# np.savetxt('datasets/temp/labels', yTest, delimiter=',')

