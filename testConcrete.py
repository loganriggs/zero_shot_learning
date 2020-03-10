from ConcreteAutoencoder import *
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU
import numpy as np

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = np.reshape(x_train, (len(x_train), -1))
# x_test = np.reshape(x_test, (len(x_test), -1))
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

x = np.loadtxt('datasets/finalProcessed/normalizeXAbs.txt',  delimiter=',')
y = np.loadtxt('datasets/finalProcessed/processedY.txt',  delimiter=',')
sampleSize = 400
ySize = np.size(y)
x2 = x[0:ySize, 0:sampleSize]
X,Y = unison_shuffled_copies(x2,y)
percentTrain = .6
trainSize = round(percentTrain * ySize)
xTrain = X[0:trainSize]
xTest = X[trainSize: ySize]
yTrain = Y[0:trainSize]
yTest = Y[trainSize:ySize]
print(np.shape(xTrain))



def decoder(x):
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(400)(x)
    return x



selector = ConcreteAutoencoderFeatureSelector(K = 20, output_function = decoder, num_epochs = 20, tryout_limit=1)

selector.fit(xTrain, xTrain, xTest, xTest)
print(selector.get_support(indices= True))
