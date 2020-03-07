import numpy as np

fileName = "superCleanyValues_Formatted.txt"

with open(fileName) as f:
    macAddress = f.readlines()

y = np.copy(macAddress)
dictIndex = 0
signalDict = {}

for index in range(len(macAddress)):
    mcA = macAddress[index]
    val = signalDict.keys()
    if(mcA not in signalDict.keys()):
        signalDict[mcA] = dictIndex
        dictIndex += 1
    y[index] = signalDict[mcA]
    print(index)

for siger in signalDict:
    print(siger)
print(signalDict)
yTemp = [int(i) for i in y]

for element in range(max(yTemp)):
    print(element, ": ", yTemp.count(element))
print("+===================================+")
with open(fileName + 'yValues.txt', 'w') as file:
    for macAdd in y:
        file.write("{0}\n".format(macAdd))

#7, 4, 0, 1, 2, 3, 8, 5, 11,
        # !6
#[0:5, 7,8,11]