import tkinter as tk
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from NeuralNetwork import *
import os
from datetime import datetime
import csv



def msg(text):
    msgWin = tk.Toplevel()
    tk.Label(msgWin, text=text).pack(padx=20, pady=20)



def getOuputPrecitionsAccuracy(images, labels):
    global output, predictions, accuracy
    _, _, _, output = forwardProp(L1W, L1B, L2W, L2B, images)
    predictions = getPredictions(output)
    accuracy = getAccuracy(predictions, labels)
    


def genParams(i, alpha):
    global L1W, L1B, L2W, L2B, paramsList, paramsExist
    L1W, L1B, L2W, L2B = gradientDescent(trainingLabels, trainingImages, testingLabels, testingImages, i, alpha, root, paramsFrame)
    paramsList = [L1W, L1B, L2W, L2B]
    paramsExist = True
    getOuputPrecitionsAccuracy(testingImages, testingLabels)
    tk.Label(paramsFrame, text=f"Using parameters: Generated", width=50).grid(row=6, column=0, padx="5px", pady="5px")
    tk.Label(paramsFrame, text=f"Accuracy: {round(accuracy, 1)}%", width=20).grid(row=7, column=0, padx="5px", pady="5px")



def loadImageData(index):
    testingData = np.array(pd.read_csv("mnist_test.csv")).T
    testingLabels = testingData[0]
    testingImages = testingData[1:testingData.size]
    return testingImages.T[index].reshape(28, 28), testingLabels[index]



def loadPrediction(index):
    relevantOutput = output.T[index]
    prediction = getPredictions(relevantOutput)

    tk.Label(dataFrame, text=f"Prediction: {prediction}", width=15, background="yellow").grid(row=1, column=0, padx="5px", pady="5px")

    predictionIndex = list(relevantOutput).index(max(relevantOutput))
    for i, probability in enumerate(relevantOutput):
        if i == predictionIndex:
            tk.Label(dataFrame, text=f"P({i}) = {round(probability*100, 1)}%", background="red", foreground="white", width=15).grid(row=2+i, column=0, padx="5px", pady="5px")
        else:
            tk.Label(dataFrame, text=f"P({i}) = {round(probability*100, 1)}%", width=15).grid(row=2+i, column=0, padx="5px", pady="5px")



def loadImage(index):
    imageIndexEntry.delete(0,tk.END)
    imageIndexEntry.insert(0,index)

    img, label = loadImageData(int(index))

    IMGax.clear()
    IMGax.imshow(img, cmap='gray')
    IMGcanvas.draw()

    tk.Label(dataFrame, text=f"Actual: {label}", width=15, background="yellow").grid(row=0, column=0, padx="5px", pady="5px")

    loadPrediction(index)



def initialImageLoad(index):
    global IMGfig, IMGax, IMGcanvas

    img, label = loadImageData(int(index))

    IMGfig, IMGax = plt.subplots(figsize=(4.5, 4.5))
    IMGcanvas = FigureCanvasTkAgg(IMGfig, master=imageFrame)
    IMGcanvas.get_tk_widget().grid(row=5, column=0, padx="5px", pady="5px", sticky="nsew")

    IMGax.imshow(img, cmap='gray')
    IMGcanvas.draw()
    
    tk.Label(dataFrame, text=f"Actual: {label}", width=15, background="yellow").grid(row=0, column=0, padx="5px", pady="5px")



def initialAccuracyGraphLoad():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    canvas = FigureCanvasTkAgg(fig, master=paramsFrame)
    canvas.get_tk_widget().grid(row=5, column=0, padx="5px", pady="5px", sticky="nsew")
    canvas.draw()



def saveParams():
    if paramsExist:
        newpath = datetime.now().strftime('params/params_created_on_%d_%m_%Y_at_%H_%M_%S')
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        i = 0
        for params in paramsList:
            if i == 0:
                filename = os.path.join(newpath, "Layer_1_weights.csv")
            elif i == 1:
                filename = os.path.join(newpath, "Layer_1_biases.csv")
            elif i == 2:
                filename = os.path.join(newpath, "Layer_2_weights.csv")
            elif i == 3:
                filename = os.path.join(newpath, "Layer_2_biases.csv")
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(params)
            i += 1
        msg("Successfully saved parameters")
    else:
        msg("You haven't generated any parameters")



def importParams(parent, win):
    global L1W, L1B, L2W, L2B
    L1W = np.genfromtxt(f'params/{parent}/Layer_1_weights.csv', delimiter=',')
    L1B = np.genfromtxt(f'params/{parent}/Layer_1_biases.csv', delimiter=',').reshape(10, 1)
    L2W = np.genfromtxt(f'params/{parent}/Layer_2_weights.csv', delimiter=',')
    L2B = np.genfromtxt(f'params/{parent}/Layer_2_biases.csv', delimiter=',').reshape(10, 1)
    getOuputPrecitionsAccuracy(testingImages, testingLabels)
    tk.Label(paramsFrame, text=f"Using parameters: {parent}", width=50).grid(row=6, column=0, padx="5px", pady="5px")
    tk.Label(paramsFrame, text=f"Accuracy: {round(accuracy, 1)}%", width=20).grid(row=7, column=0, padx="5px", pady="5px")
    win.destroy()



def loadParams():
    loadParamsWin = tk.Toplevel()
    folders = [folder for folder in os.listdir("params")]
    for i, folder in enumerate(folders):
        tk.Button(loadParamsWin, text=folder, command=lambda f=folder: importParams(f, loadParamsWin)).grid(row=i, column=0, padx="10px", pady="5px")



def initialisation():
    global paramsExist, trainingLabels, trainingImages, testingLabels, testingImages
    paramsExist = False
    trainingLabels, trainingImages, testingLabels, testingImages = getData("mnist_train.csv", "mnist_test.csv")
    initialAccuracyGraphLoad()
    initialImageLoad(0)



# Root
root = tk.Tk()
root.title("MNIST Neural Network Tester")



# Parameters Frame
paramsFrame = tk.Frame(root)
paramsFrame.grid(row=0, column=0, padx="5px", pady="5px")
iterationsFrame = tk.Frame(paramsFrame)
iterationsFrame.grid(row=0, column=0, padx="5px", pady="5px")
alphaFrame = tk.Frame(paramsFrame)
alphaFrame.grid(row=1, column=0, padx="5px", pady="5px")
saveLoadFrame = tk.Frame(paramsFrame)
saveLoadFrame.grid(row=3, column=0, padx="5px", pady="5px")

tk.Label(iterationsFrame, text="Iterations: ").grid(row=0, column=0, padx="5px")
iterationsEntry = tk.Entry(iterationsFrame)
iterationsEntry.grid(row=0, column=1, padx="5px")

tk.Label(alphaFrame, text="Learning Rate: ").grid(row=0, column=0, padx="5px")
alphaEntry = tk.Entry(alphaFrame)
alphaEntry.grid(row=0, column=1, padx="5px")

tk.Button(paramsFrame, text="Generate Parameters", command=lambda: genParams(int(iterationsEntry.get()), float(alphaEntry.get()))).grid(row=2, column=0, padx="5px", pady="5px")

tk.Button(saveLoadFrame, text="Save Parameters", command=saveParams).grid(row=0, column=0, padx="5px")
tk.Button(saveLoadFrame, text="Load Parameters", command=loadParams).grid(row=0, column=1, padx="5px")



# Image Frame
imageFrame = tk.Frame(root)
imageFrame.grid(row=0, column=1, padx="5px", pady="5px")

tk.Label(imageFrame, text="Image Index:").grid(row=0, column=0, padx="5px", pady="5px")

imageIndexFrame = tk.Frame(imageFrame)
imageIndexFrame.grid(row=1, column=0, padx="5px", pady="5px")

tk.Button(imageIndexFrame, text="<<", command=lambda: loadImage(int(imageIndexEntry.get())-1)).grid(row=0, column=0, padx="5px", pady="5px")

imageIndexEntry = tk.Entry(imageIndexFrame)
imageIndexEntry.grid(row=0, column=1, padx="5px", pady="5px")
imageIndexEntry.insert(0, "0")

tk.Button(imageIndexFrame, text=">>", command=lambda: loadImage(int(imageIndexEntry.get())+1)).grid(row=0, column=2, padx="5px", pady="5px")

tk.Button(imageFrame, text="Load Image", command=lambda: loadImage(int(imageIndexEntry.get()))).grid(row=2, column=0, padx="5px", pady="5px")



# Data Frame
dataFrame = tk.Frame(root)
dataFrame.grid(row=0, column=2, padx="5px", pady="5px")



initialisation()
root.mainloop()