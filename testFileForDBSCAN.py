import numpy as np
from dbscan import *
from plot3d import *

encodedImgs = np.loadtxt('datasets/temp/encodedImgs', delimiter=',')
encodedImgsTrain = np.loadtxt('datasets/temp/encodedImgsTrain', delimiter=',')
labels = np.loadtxt('datasets/temp/labels', delimiter=',')
labels = labels.astype(int)

epsilon = 0.015
minSamps = 2

print("Training")
dbscan(encodedImgsTrain, labels, epsilon = epsilon, minSamples=minSamps)
print("Training")
dbscan(encodedImgs, labels, epsilon = epsilon, minSamples=minSamps)
