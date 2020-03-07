import numpy as np
tempX = []
tempY = []
sampleSize = 800
classSize = 10
numberOfSamples = 100
for count in range(classSize):
    tempX = tempX + [([count**2]*sampleSize + np.random.rand(sampleSize)).tolist() for i in range(0, numberOfSamples)]
    tempY = tempY + [count]*numberOfSamples

np.savetxt('toyX', tempX, delimiter=',')
np.savetxt('toyY', tempY, delimiter=',')
