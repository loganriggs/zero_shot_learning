import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot3D(xValues, labels, figure = 0):
    minClass = min(labels)
    numberOfClasses = int(max(labels) - minClass)

    fig = plt.figure(figure)
    ax = plt.axes(projection='3d')
    colors = ["r", "b", "y", "c", "m"]
    for i in range(numberOfClasses+1):
        classLocation = np.argwhere(labels == i+minClass)
        ax.scatter3D(xValues[classLocation, 0], xValues[classLocation, 1], xValues[classLocation, 2]) #3D
