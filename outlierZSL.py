import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

xTrainAvg = np.loadtxt('zeroShotDataSet/xTrainAvg',  delimiter=',')
xTrainVar = np.loadtxt('zeroShotDataSet/xTrainVar', delimiter=',')

xTestPredict = np.loadtxt('zeroShotDataSet/xTestPredict',  delimiter=',')
yTest = np.loadtxt('zeroShotDataSet/yTest',  delimiter=',')

xUnknownClassPredict = np.loadtxt('zeroShotDataSet/xUnknownClassPredict',  delimiter=',')
yUnknownClass = np.loadtxt('zeroShotDataSet/randClassY', delimiter=',')

# toyX = np.loadtxt('toyX',  delimiter=',')
# toyY = np.loadtxt('toyY', delimiter=',')
# clustering = DBSCAN(eps = 0.4, min_samples=2).fit(toyX[:,0:3])
# for one, two in zip(toyY, clustering.labels_):
#     print("yTest: ", one, " | DBSCAN: ", two)
# print(np.shape(toyX))
# print(toyX[:,0])
# #

#Possibly try PCA here?

pca = PCA(.90)
xPCA = pca.fit_transform(xUnknownClassPredict)
print(np.shape(xPCA))
#PCA = .95, eps = 1, works for all classes xTestPredict.
#Reversing to_categorical
yTest2 = [np.argmax(y, axis=None, out=None) for y in yTest]

print(np.shape(yTest2))
minClass = min(yTest2)
numUnknownClasses = int(max(yTest2) - minClass)
confusionMatrix = np.zeros((numUnknownClasses+1, numUnknownClasses+1))
outliersError = 0

print(np.bincount(yTest2))
clustering = DBSCAN(eps = 0.2, min_samples=3).fit(xTestPredict)
for one, two in zip(yTest2, clustering.labels_):
    if (two < numUnknownClasses and two != -1):
        confusionMatrix[int(one - minClass), two] += 1
    else:
        outliersError += 1
    # print("yTest: ", one, " | DBSCAN: ", two)

print("Confusion Matrix. Correct class is the greatest in a row")
print(confusionMatrix)
print("Outliers Error(-1 or more classes than actual classes): ", outliersError)

minClass = min(yUnknownClass)
numUnknownClasses = int(max(yUnknownClass) - minClass)
confusionMatrix = np.zeros((numUnknownClasses+1, numUnknownClasses+1))
outliersError = 0
print(np.bincount(yUnknownClass.astype(int)))
clustering = DBSCAN(eps=0.15, min_samples=20).fit(xPCA)
for one, two in zip(yUnknownClass, clustering.labels_):
    if(two < numUnknownClasses and two!= -1):
        confusionMatrix[int(one-minClass),two] +=1
    else:
        outliersError +=1
    # print("yUnknown: ", one, " | DBSCAN: ", two)
print("Confusion Matrix. Correct class is the greatest in a row")
print(confusionMatrix)
print("Outliers Error(-1 or more classes than actual classes): ", outliersError)

#Test outlier accuracy (????)
#Build array of outliers using the two arrays
#Google outlier detection multivariate gaussian
#1. Signals in xTestPredict should NOT be an outlier

#2. Signals in xUnknownClassPredict SHOULD be an outlier


#DBSCAN
#Dbscan on outliers. (May be more useful with 2 unknown classes) Test accuracy

#TODO: Maybe do push idea. This may not involve outlier detection afterwards...