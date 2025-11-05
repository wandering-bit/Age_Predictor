import numpy as np
import matplotlib.pyplot as plt
import time

weights = 10
bias = 1000
x=[10,30   ,45   ,25   ,78   ,34   ,64   ,27   ,76   ,34   ,54   ,76   ,23   ,54   ,12,35   ]
y=[0 ,40000,45000,30000,80000,42000,70000,33000,55000,78000,24000,60000,26000,66000,0 ,43000]
learningRate = 0.00002

def CalculateOutput(input,weights, bias):
    output = (input*weights)+bias
    return output

def CalculateError(expected,predicted):
    return (float(expected)-float(predicted))

def CalculateWeightGradient(x,predicted,expected):
    return -2 * x * (expected - predicted)

def CalculateBiasGradient(predicted,expected):
    return (-2 * (expected - predicted))

def UpdateWeight(gradient):
    global weights
    weights = weights-(learningRate*gradient)

def UpdateBias(gradient):
    global bias
    bias = bias-(learningRate*gradient)

def FeedForward(x,y,epoch):
    for i in range (0,epoch):
        print(i+1,"/",epoch)
        avgWeightGradient = 0.0
        avgBiasGradient = 0.0
        for i in range(0,len(x)):
            predictedOutput = CalculateOutput(x[i],weights,bias)
            weightGradient = CalculateWeightGradient(x[i],predictedOutput,y[i])
            biasGradient = CalculateBiasGradient(predictedOutput,y[i])
            avgWeightGradient+=weightGradient
            avgBiasGradient +=biasGradient
        UpdateWeight(avgWeightGradient/len(x))
        UpdateBias(avgBiasGradient/len(x))
        DrawLine(x,y)
        DrawScatter(x,y,"age","salary")
        plt.draw()
        plt.pause(0.01)
        plt.clf()

def DrawScatter(x,y,xLabel,yLabel):
    plt.scatter(x,y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

def DrawLine(x,y):
    lineXMin = min(x)
    lineXMax = max(x)
    xLine =[]
    yLine =[]
    xLine.append(lineXMin)
    yLine.append((weights*lineXMin)+bias)
    xLine.append(lineXMax)
    yLine.append((weights*lineXMax)+bias)
    plt.plot(xLine,yLine)

def Predict(x):
    return (x*weights)+bias

FeedForward(x,y,2000)

averageError = 0
for i in range(0,len(x)):
    predicted = (int(Predict(x[i])))
    averageError += CalculateError(y[i],predicted)
print("average error is ", averageError/len(x))