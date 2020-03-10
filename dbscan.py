from sklearn.cluster import DBSCAN
import numpy as np

def dbscan(x, y, epsilon, minSamples = 3, verbose = 0):
    minClass = min(y)
    numClasses = int(max(y) - minClass)
    confusionMatrix = np.zeros((numClasses+1, numClasses+1))
    outliersError = 0
    overClassError = 0
    print("Label Count:")
    print(np.bincount(y))
    clustering = DBSCAN(eps = epsilon, min_samples=minSamples).fit(x)
    for one, two in zip(y, clustering.labels_):
        if (two <= numClasses and two != -1):
            confusionMatrix[int(one - minClass), two] += 1
        elif (two == -1):
            outliersError += 1
        else:
            overClassError += 1
        if(verbose ==1):
            print("label: ", one, " | DBSCAN: ", two)

    print("Confusion Matrix. \"Correct\" class is the greatest in a row")
    print(confusionMatrix)
    print("Outliers Error: ", outliersError)
    print("OverClass Error: ", overClassError)