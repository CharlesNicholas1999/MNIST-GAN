# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:42:03 2022

@author: q89422cn
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]=""

#%% Global Variables

# variables about real image (RI)
datadim = [28,28] # RI pixel dimension
datalen = datadim[0]*datadim[1] #total number of pixels
realImageNum = 3 # drawn number number in pixel image
quantity = 2 # number real images

# Network Variables
learningRate = 0.25 #training learningRate
DisLayers = [datalen, 1] # Network Layout of Discriminator
GenLayers = [1, 5, datalen] # Network Layout of Generator

# Output Variables
numIterations = 500 # number of training epochs conducted
plotMods = [1, 2, 5] # defines which images are plotted (MSD of each power of 10)

# Random Seed
tf.random.set_seed(20)
np.random.seed(20)


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

class RealImageGenerator():
    def __init__(self, name = None):
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        
        self.x_train = self.x_train.reshape(60000,-1).astype("float32") /255
        self.x_test = self.x_test.reshape(10000,-1).astype("float32") /255
    
    def getSingleNumImages(self, num, quantity, isTrain):
        self.Images = []
        self.ImageValues = []
        
        if isTrain:
            DataQuantity = 60000
            ImageData = self.x_train
            NumData = self.y_train
        else:
            DataQuantity = 10000
            ImageData = self.x_test
            NumData = self.y_test
        
        count = 0
        idx = list(range(DataQuantity))
        
        for i in idx:
            if (NumData[i] == num):
                self.Images.append([ImageData[i].numpy()])
                self.ImageValues.append([NumData[i].numpy()])
                count +=1
                if (count == quantity):
                    return self.Images
                

#%% Discriminator

class Discriminator():
    def __init__(self, LayerSequence, name = None):
        
        #define discriminator model
        self.model = keras.Sequential()
        
        #add the input
        self.model.add(keras.Input(shape = LayerSequence[0]))
        
        #add the internal layers
        for layer in LayerSequence[0:-1]:
            self.model.add(layers.Dense(layer, activation = 'relu'))
            
        #add the output layer
        self.model.add(layers.Dense(LayerSequence[-1], activation = 'softmax'))
        
        #print the model layer structure
        print(self.model.summary())
        #keras.utils.plot_model(self.model, "Discirminator Model.png", show_shapes=True)
        
        # define the training parameters for the model
        self.model.compile(
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits = False),
            optimizer = keras.optimizers.Adam(lr = 0.001),
            metrics = ["accuracy"],
        )
        
#%% Generator

class Generator():
    def __init__(self, LayerSequence, name = None):
        
        # define the model inputs
        inputs = keras.Input(shape = LayerSequence[0])
        
        # create the model internal layers
        modelLayer = inputs

        for layer in LayerSequence[1:-1]:
            modelLayer = layers.Dense(layer, activation = 'relu')(modelLayer)
        
        #define the model outputs
        outputs = layers.Dense(LayerSequence[-1], activation = 'softmax')(modelLayer)
        
        # create the generator model
        self.model = keras.Model(inputs = inputs, outputs = outputs)
        
        #print the model later structure
        print(self.model.summary())
        #keras.utils.plot_model(self.model, "Generator Model.png", show_shapes=True)
        
        # define the training parameters
        self.model.compile(
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits = False),
            optimizer = keras.optimizers.Adam(lr = 0.001),
            metrics = ["accuracy"],
        )
        
        
#%% main

def main():
    Data = RealImageGenerator()
    
    Dis = Discriminator(DisLayers)
    
    Gen = Generator(GenLayers)
    
main()






















