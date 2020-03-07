import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

xFileName = "Abs"

y = np.loadtxt("superCleanYValues.txt")
y = y.astype(int)
x = np.loadtxt("superClean"+ xFileName + "X.txt")


maxSize = np.size(y)
#select Classes with ~100 examples
selectedLocations = (y < 6) | (y == 7) | (y == 8) | (y == 11)
processedY = y[selectedLocations]
processedX = x[selectedLocations]
print(np.bincount(processedY))
processedY[processedY==7] -= 7-6
processedY[processedY==8] -= 8-7
processedY[processedY==11] -= 11-8
print(np.bincount(processedY))

#normalize
plt.plot(processedX[1,:])
plt.plot(processedX[0,:])
# newX = preprocessing.scale(processedX, 1)
newX = preprocessing.normalize(processedX)
plt.figure()
plt.plot(newX[1,:])
plt.plot(newX[0,:])
# plt.show()
np.savetxt("datasets/finalProcessed/normalizeX" + xFileName + ".txt",newX , delimiter=',')
np.savetxt("datasets/finalProcessed/processedY.txt", processedY, delimiter=',')