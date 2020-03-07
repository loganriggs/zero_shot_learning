from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

startAt = 0
sampleSize = 800
startShape = sampleSize - startAt


deep = True
secondLayer = 200
thirdLayer = 50
fourthLayer = 10
# fifthLayer = 10
middleLayer = 2
# this is our input placeholder
input_img = Input(shape=(startShape,))
regularizerValue = 10e-6
#------------------ No 0      No1     No2      No3          No4       No5      No6     No7          No8              No9          No10
activationNames = ["softmax", "elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "exponential", "linear"]
ind = 10
activationFunction = activationNames[10]
decodedActivation = activationNames[7]
outerActivation = activationNames[5]
innerActivation = activationNames[10]

if(deep):
    encoded = Dense(secondLayer, activation=activationFunction)(input_img)
    encoded = Dense(thirdLayer, activation=activationFunction)(encoded)
    encoded = Dense(fourthLayer, activation=activationFunction)(encoded)
    encoded = Dense(middleLayer, activation=activationFunction)(encoded)

    # encoded = Dense(secondLayer, activation=outerActivation)(input_img)
    # encoded = Dense(thirdLayer, activation=innerActivation)(encoded)
    # ------------------------------------------
    # encoded = Dense(secondLayer)(input_img)
    # encoded = Dense(thirdLayer)(encoded)
    #---------------------------------------------------
    # encoded = Dense(middleLayer)(encoded)


    # encoded = Dense(secondLayer, activation=activationFunction,
    #     activity_regularizer = regularizers.l1(regularizerValue))(input_img)
    # encoded = Dense(thirdLayer, activation=activationFunction,
    #     activity_regularizer = regularizers.l1(regularizerValue*.01))(encoded)
    # encoded = Dense(middleLayer, activation=activationFunction)(encoded)


    decoded = Dense(fourthLayer, activation=activationFunction)(encoded)
    decoded = Dense(thirdLayer, activation=activationFunction)(encoded)
    decoded = Dense(secondLayer, activation=activationFunction)(encoded)
    # decoded = Dense(startShape, activation=decodedActivation)(decoded)
    # decoded = Dense(thirdLayer, activation=innerActivation)(encoded)
    # decoded = Dense(secondLayer, activation=outerActivation)(encoded)
    # -----------------------------------------------------------
    # decoded = Dense(thirdLayer)(encoded)
    # decoded = Dense(secondLayer)(encoded)
    #------------------------------------------------------------
    decoded = Dense(startShape, activation=decodedActivation)(decoded)
else:
    encoded = Dense(secondLayer, activation=activationFunction)(input_img)
    decoded = Dense(startShape, activation=decodedActivation)(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(startShape,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
# decoder = Model(encoded_input, decoder_layer(encoded_input))
# autoencoder.compile(optimizer='adadelta', loss='mean_squared_error') #0.008
# autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error') #0.006
# autoencoder.compile(optimizer='adadelta', loss='mean_absolute_percentage_error') #0.36
# autoencoder.compile(optimizer='adadelta', loss='mean_squared_logarithmic_error') #0.0043
# autoencoder.compile(optimizer='adadelta', loss='squared_hinge') #0.59
# autoencoder.compile(optimizer='adadelta', loss='hinge') #0.73
# autoencoder.compile(optimizer='adadelta', loss='categorical_hinge') #0.0013 ****
# autoencoder.compile(optimizer='adadelta', loss='logcosh') #0.0041
#Different Optimizer, loss?
# autoencoder.compile(optimizer='sgd', loss='mean_squared_logarithmic_error') #0.0029
# autoencoder.compile(optimizer='RMSprop', loss='mean_squared_logarithmic_error') #0.0031
# autoencoder.compile(optimizer='Adagrad', loss='mean_squared_logarithmic_error') #0.0027
# autoencoder.compile(optimizer='Adam', loss='mean_squared_logarithmic_error') #0.0028
autoencoder.compile(optimizer='Adamax', loss='mean_squared_logarithmic_error') #0.0015* 2.1848e-04
# autoencoder.compile(optimizer='Nadam', loss='mean_squared_logarithmic_error') #0.0014



# x = np.loadtxt("400fftSignal.txt")
# y = np.loadtxt("fixedy.txt")

y = np.loadtxt("WiFiData1yValues.txt")
x1 = np.loadtxt("WiFiData1400RealFFT.txt")
x2 = np.loadtxt("WiFiData1400ComplexFFT.txt")
x = np.ones((np.shape(x1)[0], 800))
x[:, 0:800:2] = x1
x[:, 1:800:2] = x2

#Begin Toy dataset
# x = np.loadtxt('toyX',  delimiter=',')
# y = np.loadtxt('toyY', delimiter=',')
# sampleSize = 800
#End toy dataset

ySize = np.size(y)
x2 = x[0:ySize, startAt:sampleSize]

#Begin Normalizing Coe
# varX2 = np.var(x2,1)
# print("shape shoul be 621, not sampleSize: ", np.shape(varX2))
# # x2/sqrt(var)
#
# x2Norm = x2/np.max(x2)

# x2 = x2Norm
#End Normalizing Code

X,Y = unison_shuffled_copies(x2,y)
# X = x2
# Y = y
percentTrain = .6
trainSize = round(percentTrain * ySize)
xTrain = X[0:trainSize]
xTest = X[trainSize: ySize]
yTrain = Y[0:trainSize]
yTest = Y[trainSize:ySize]
print(np.shape(xTrain))

# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)
# print(x_test.shape)

autoencoder.fit(xTrain, xTrain,
                epochs=20,
                batch_size=10,
                shuffle=True,
                validation_data=(xTest, xTest))

# encode and decode some digits
# note that we take them from the *test* set
nClasses = 5
colors = ["r", "b", "y", "c", "m"]
encoded_imgs = encoder.predict(xTest)
encoded_imgs_train = encoder.predict(xTrain)
encoded_imgs2 = autoencoder.predict(xTest)
n=100
plt.plot(xTest[n,:],  'r-')
plt.plot(encoded_imgs2[n,:],  'b-')
plt.show()

clustering = DBSCAN(eps = 1.00, min_samples=4).fit(encoded_imgs_train)
for one, two in zip(yTrain, clustering.labels_):
    print("yTest: ", one, " | DBSCAN: ", two)

for i in range(nClasses):
    classLocation = np.argwhere(yTest == i)
    plt.plot(encoded_imgs[classLocation, 0], encoded_imgs[classLocation, 1],  colors[i]+'s')
plt.show()
for i in range(nClasses):
    classLocation = np.argwhere(yTrain == i)
    plt.plot(encoded_imgs_train[classLocation, 0], encoded_imgs_train[classLocation, 1],  colors[i]+'s')
plt.show()


# decoded_imgs = decoder.predict(encoded_imgs)

#
# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()