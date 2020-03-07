from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

accList = []
trainSizeList = []
lossList = []
rbfSizeList = []
runType = "complex"
fileName = "WiFiData1"

# y = np.loadtxt(fileName + "yValues.txt")
y = np.loadtxt("WiFiData1yValues.txt")


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


#Begin PCA
maxSize = np.size(y)
y = y.astype(int)

pca = PCA(.90)
tempX = x[0:maxSize,0:sampleSize]
principalComponents = pca.fit_transform(tempX)
x2 = principalComponents
print(np.shape(x2))

# print("Xshape: ", np.shape(x2))
# print("Yshape: ", np.shape(y))
# randClassLocation = np.where(y >= 3)
# restOfClassLocation = [i for i in range(maxSize) if i not in randClassLocation[0]]
# randClassY = y[randClassLocation]
# randClassX = x2[randClassLocation]
# y = y[restOfClassLocation]
# x2 = x2[restOfClassLocation]
# print("Xshape: ", np.shape(x2), " + ", np.shape(randClassX))
# print("Yshape: ", np.shape(y), " + ", np.shape(randClassY))

# X,Y = unison_shuffled_copies(x2,y)

# percentTrain = 0.8
# trainSize = round(percentTrain * maxSize)

# xTrain = X[0:trainSize]
# xTest = X[trainSize: maxSize]
# yTrain = Y[0:trainSize]
# yTest = Y[trainSize:maxSize]
numClasses = max(y)+1
xTrainAvg = [[] for arr in range(numClasses)]
for i in range(numClasses):
    classLocation = np.argwhere(y == i)
    xTrainAvg[i]= np.average(x2[classLocation,:] ,0)[0]

print("Average: ")
print(xTrainAvg)
#End PCA
#Begin DBSCAN
n = 1
epsilon = 3
for epsilonIter in range(n):
    epsilon*=1.5
    minSample = 2
    for minSamplesIter in range(n):
        minSample +=1
        correctCount = 0
        # print("___________+++++++++++++++++++++______-")
        clustering = DBSCAN(eps = epsilon, min_samples=minSample).fit(x2)
        for one, two in zip(y, clustering.labels_):
            print("yTest: ", one, " | DBSCAN: ", two)
            if(one == two):
                correctCount +=1
        print("Epsilon:     ", epsilon, " | minSample: ", minSample)
        print("Correct: ", correctCount, "/", maxSize)

# print("===========================================================")
# print("unknown classes X PCA: ")
# clustering = DBSCAN(eps=0.1, min_samples=4).fit(randClassX)
# for one, two in zip(randClassY, clustering.labels_):
#     print("yUnknown: ", one, " | DBSCAN: ", two)
#
