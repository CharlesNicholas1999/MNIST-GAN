# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:42:03 2022

@author: q89422cn
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "True"

#%% Global Variables

# variables about real image (RI)
DATADIM = [28,28] # RI pixel dimension
DATALEN = DATADIM[0]*DATADIM[1] #total number of pixels
REALIMAGENUM = 3 # drawn number

# Network Variables
LEARNINGRATE = 0.001 #training learningRate
BATCHSIZE = 32

# GAN Variables
DISLAYERS = [DATALEN, 1] # Network Layout of Discriminator
GENLAYERS = [1, 5, DATALEN] # Network Layout of Generator

# Output Variables
NUMITERATIONS = 500 # number of training epochs conducted

# Plotting Variables
PLOTMODS = [1, 2, 5] # defines which images are plotted (MSD of each power of 10)

# Random Seed
tf.random.set_seed(20)


#%% Plotting

# Draws a set of Grid Data on a single figure
def GridPlotter(data, datadim, rows, cols, title, subtitle):
    fig, axes = plt.subplots(nrows = rows, ncols = cols, layout = 'constrained', sharey=True, sharex=True)
    fig.suptitle(title, fontsize = 20)
    
    if len(data) == 1:
        axes.imshow(tf.reshape(data, datadim), cmap = "Greys", interpolation = 'nearest', vmin = 0, vmax = 1)
    else:
        for ax, image, title in zip(axes.flatten(), data, subtitle):
                ax.set_title(title, fontsize = 5)
                ax.yaxis.set_visible(False)
                ax.xaxis.set_visible(False)
                ax.imshow(tf.reshape(image, datadim), cmap = "Greys", interpolation = 'nearest', vmin = 0, vmax = 1)

# Plots sets of continous data on 1 figure
def DataPlotter(xdata, ydata, rows, cols, title, subtitles):
    fig, axes = plt.subplots(nrows = rows, ncols = cols, layout = 'constrained', sharey=True, sharex=True)
    fig.suptitle(title, fontsize = 20)
    
    for ax, x, y, subtitle in zip(axes.flatten(), xdata, ydata, subtitles):
        ax.set_title(subtitle, fontsize = 10)
        ax.yaxis.set_visible(True)
        ax.xaxis.set_visible(True)
        ax.plot(x, y)

# returns the number of digits of its input
def numDigits(epoch):
    temp = epoch + 1
    numDigits = 1
    while temp >= 10:
        temp = np.floor_divide(temp, 10)
        numDigits += 1
    return numDigits
   
# determines if current epoch should be plotted     
def ShouldPlot(epoch):
    mods = np.array(plotMods)*10**(numDigits(epoch)-1)
    for i in mods:
        if (epoch+1) == i:
            return True
    return False

# returns an appropriate number of plot dimensions for 
def PlotDim(n):
    if n <= 5:
        return [1, n]
    
    maxnum = int(np.ceil(np.sqrt(n)))
    for i in range(maxnum, 1, -1):
            if n % i == 0:
                if (i > 8) or (n / i) > 8:
                    break
                if i > (n/i):
                    return[int(n/i),i]
                else:
                    return [i, int(n/i)]
            
    return [int(np.ceil(n/5)), 5]

#%% obtain real data

def GenRealImages(num, BatchSize):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(-1, DATALEN).astype("float32") / 255
    x_test = x_test.reshape(-1, DATALEN).astype("float32") / 255
    
    trainData = []
    testData = []
    
    for idx, data in enumerate(x_train):
        if y_train[idx] == num:
            trainData.append(data)
    trainDataLen = len(trainData) - (len(trainData) % BatchSize)
    
    for idx, data in enumerate(x_test):
        if y_test[idx] == num:
            testData.append(data)
    testDataLen = len(testData) - (len(testData) % BatchSize)
    
    return [tf.random.shuffle(tf.constant(trainData[0:trainDataLen])), tf.random.shuffle(tf.constant(testData[0:testDataLen]))]

#%% Create a network model

def CreateModel(LayerSequence, name):
    model = keras.Sequential(name = name)
    
    model.add(keras.Input(shape = LayerSequence[0]))
    
    for layer in LayerSequence[1:-1]:
        model.add(layers.Dense(layer, activation = "relu"))
        
    model.add(layers.Dense(LayerSequence[-1], activation = "sigmoid"))
    
    return model


#%% Main

def main():
    [trainData, testData] = GenRealImages(REALIMAGENUM, BATCHSIZE)
    
    DiscModel = CreateModel(DISLAYERS, "Discrimator")
    GenModel = CreateModel(GENLAYERS, "Generator")
    
    DiscModel.summary()
    GenModel.summary()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
main()
