# L1 = Layer 1
# L2 = Layer 2
# W = Weights
# B = Biases
# Z = Unactivated
# A = Activated
# alpha = learning rate
# i = iterations

import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def getData(traingDataPath, testingDataPath):
    trainingData = np.array(pd.read_csv(traingDataPath)).T
    trainingLabels = trainingData[0]
    trainingImages = trainingData[1:trainingData.size] / 255

    testingData = np.array(pd.read_csv(testingDataPath)).T
    testingLabels = testingData[0]
    testingImages = testingData[1:testingData.size] / 255

    return trainingLabels, trainingImages, testingLabels, testingImages

def initParams():
    L1W = np.random.rand(10, 784) - 0.5
    L1B = np.random.rand(10, 1) - 0.5
    L2W = np.random.rand(10, 10) - 0.5
    L2B = np.random.rand(10, 1) - 0.5
    return L1W, L1B, L2W, L2B

def ReLU(Z):
    A = np.maximum(0, Z)
    return A

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forwardProp(L1W, L1B, L2W, L2B, inputs):
    L1Z = np.dot(L1W, inputs) + L1B
    L1A = ReLU(L1Z)
    L2Z = np.dot(L2W, L1A) + L2B
    L2A = softmax(L2Z)
    return L1Z, L1A, L2Z, L2A

def oneHotEncode(labels):
    labelsList = list(labels)
    oneHotLabels = []
    for label in labelsList:
        oneHotLabel = ([0] * 10)
        oneHotLabel[label] = 1
        oneHotLabels.append(oneHotLabel)
    return np.array(oneHotLabels).T

def backProp(L1Z, L1A, L2A, L2W, inputs, oneHotLabels):
    numOfImages = inputs.shape[1]

    dL2Z = L2A - oneHotLabels
    dL2W = np.dot(dL2Z, L1A.T) / numOfImages
    dL2B = np.sum(dL2Z, axis=1, keepdims=True) / numOfImages

    dL1A = np.dot(L2W.T, dL2Z)
    dL1Z = dL1A * (L1Z > 0)
    dL1W = np.dot(dL1Z, inputs.T) / numOfImages
    dL1B = np.sum(dL1Z, axis=1, keepdims=True) / numOfImages

    return dL1W, dL1B, dL2W, dL2B

def updateParams(L1W, L1B, L2W, L2B, dL1W, dL1B, dL2W, dL2B, alpha):
    L1W -= alpha * dL1W
    L1B -= alpha * dL1B
    L2W -= alpha * dL2W
    L2B -= alpha * dL2B
    return L1W, L1B, L2W, L2B

def getPredictions(output):
    return np.argmax(output, axis=0)

def getAccuracy(predictions, labels):
    numCorrect = np.sum(predictions == labels)
    percentageCorrect = (numCorrect / labels.shape[0]) * 100
    return percentageCorrect

def updateData(x, y, ax, canvas, root):
    ax.clear()
    ax.plot(x, y)
    canvas.draw()
    root.update()

def gradientDescent(trainingLabels, trainingImages, testingLabels, testingImages, i, alpha, root, graphContainer):

    X_iterations = []
    Y_accuracy = []

    L1W, L1B, L2W, L2B = initParams()
    oneHotLabels = oneHotEncode(trainingLabels)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    canvas = FigureCanvasTkAgg(fig, master=graphContainer)
    canvas.get_tk_widget().grid(row=5, column=0, padx="5px", pady="5px", sticky="nsew")
    canvas.draw()
    
    for j in range(i):

        L1Z, L1A, L2Z, L2A = forwardProp(L1W, L1B, L2W, L2B, trainingImages)
        dL1W, dL1B, dL2W, dL2B = backProp(L1Z, L1A, L2A, L2W, trainingImages, oneHotLabels)
        L1W, L1B, L2W, L2B = updateParams(L1W, L1B, L2W, L2B, dL1W, dL1B, dL2W, dL2B, alpha)

        _, _, _, testingDataOutput = forwardProp(L1W, L1B, L2W, L2B, testingImages)
        testingDataPredictions = getPredictions(testingDataOutput)
        testingDataAccuracy = getAccuracy(testingDataPredictions, testingLabels)

        X_iterations.append(j)
        Y_accuracy.append(testingDataAccuracy)

        updateData(X_iterations, Y_accuracy, ax, canvas, root)
            
    return L1W, L1B, L2W, L2B