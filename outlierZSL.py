import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
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

pca = PCA(3)
xPCA = pca.fit_transform(xUnknownClassPredict)
print(np.shape(xPCA))
print(np.shape(xUnknownClassPredict))
#PCA = .95, eps = 1, works for all classes xTestPredict.
#Reversing to_categorical
yTest2 = [np.argmax(y, axis=None, out=None) for y in yTest]


print(np.shape(yTest2))
minClass = min(yTest2)
numClasses = int(max(yTest2) - minClass)
confusionMatrix = np.zeros((numClasses+1, numClasses+1))
outliersError = 0
overClassError = 0
print(np.bincount(yTest2))
clustering = DBSCAN(eps = 0.6, min_samples=5).fit(xTestPredict)
for one, two in zip(yTest2, clustering.labels_):
    if (two <= numClasses and two != -1):
        confusionMatrix[int(one - minClass), two] += 1
    elif (two == -1):
        outliersError += 1
    else:
        overClassError += 1
    # print("yTest: ", one, " | DBSCAN: ", two)

print("Confusion Matrix. Correct class is the greatest in a row")
print(confusionMatrix)
print("Outliers Error: ", outliersError)
print("OverClass Error: ", overClassError)

#UNKNOWN Classes
minClass = min(yUnknownClass)
numUnknownClasses = int(max(yUnknownClass) - minClass)
confusionMatrix = np.zeros((numUnknownClasses+1, numUnknownClasses+1))
outliersError = 0
overClassError = 0
print(np.bincount(yUnknownClass.astype(int)))
print("classes: ", numUnknownClasses)

#PLOT Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
yPlot = np.array(yTest2)
colors = ["r", "b", "y", "c", "m"]
# for i in range(numClasses+1):
for i in range(numUnknownClasses+1):
    classLocation = np.argwhere(yUnknownClass == i+minClass)
    # classLocation = np.argwhere(yPlot == i)
    # plt.plot(xPCA[classLocation, 0], xPCA[classLocation, 1],  colors[i]+'s') #2D
    ax.scatter3D(xPCA[classLocation, 0], xPCA[classLocation, 1], xPCA[classLocation, 2], colors[i]+'s') #3D
plt.show()

clustering = DBSCAN(eps=0.08, min_samples=8).fit(xPCA)
for one, two in zip(yUnknownClass, clustering.labels_):
    if(two <= numUnknownClasses and two!= -1):
        confusionMatrix[int(one-minClass),two] +=1
    elif(two == -1):
        outliersError +=1
    else:
        overClassError +=1
    # print("yUnknown: ", one, " | DBSCAN: ", two)
print("Confusion Matrix. Correct class is the greatest in a row")
print(confusionMatrix)
print("Outliers Error: ", outliersError)
print("OverClass Error: ", overClassError)


# #forLoop
# print(np.bincount(yUnknownClass.astype(int)))
# print("classes: ", numUnknownClasses)
# epsLoops = 10
# minSampleLoops = 10
# epsilon = 0.01
# epsilonPlus = 0.02
# samplePlus = 2
# for epsL in range(epsLoops):
#     epsilon += epsilonPlus
#     minSamples = 0
#     for sampleL in range(minSampleLoops):
#         minSamples += samplePlus
#         confusionMatrix = np.zeros((numUnknownClasses + 1, numUnknownClasses + 1))
#         outliersError = 0
#         overClassError = 0
#         clustering = DBSCAN(eps=epsilon, min_samples=minSamples).fit(xPCA)
#         for one, two in zip(yUnknownClass, clustering.labels_):
#             if (two <= numUnknownClasses and two != -1):
#                 confusionMatrix[int(one - minClass), two] += 1
#             elif (two == -1):
#                 outliersError += 1
#             else:
#                 overClassError += 1
#                 # print("yUnknown: ", one, " | DBSCAN: ", two)
#         print("eps: ", epsilon, " | minSamples: ", minSamples)
#         print("Confusion Matrix. Correct class is the greatest in a row")
#         print(confusionMatrix)
#         print("Outliers Error: ", outliersError)
#         print("OverClass Error: ", overClassError)
#         print("++==============================================++")

#Test outlier accuracy (????)
#Build array of outliers using the two arrays
#Google outlier detection multivariate gaussian
#1. Signals in xTestPredict should NOT be an outlier

#2. Signals in xUnknownClassPredict SHOULD be an outlier


#DBSCAN
#Dbscan on outliers. (May be more useful with 2 unknown classes) Test accuracy

#TODO: Maybe do push idea. This may not involve outlier detection afterwards...