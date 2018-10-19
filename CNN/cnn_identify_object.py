# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:51:28 2018

@author: Mahbub - A - Rob
"""

# Initialize NN
# We can initialize NN either as a sequence or as a graph
from keras.models import Sequential 

# Convolution is sequential
# Since we are going to work on images which are 2 dimentional unlike videos (time)
# API update use Conv2D instead of Convolution2D
from keras.layers import Conv2D 

# Step 2 Pooling step
from keras.layers import MaxPooling2D

# Step 3 Flatten
from keras.layers import Flatten

# Add fully connected layers in an ANN
from keras.layers import Dense


# Initializing the CNN
classifier = Sequential()

# Step 1 Convolution - CPU version
# (32, 3, 3) means 32 feture detectors, 3by3 dimensions
# Input shape is shpae of input image, black and white is a 2D image, colored are 3d image
# Tensorflow backend use input_shape = (64, 64, 3) 
# Theano backend use input_shape = (3, 64, 64)
# If there is any negative pixels in the feature map, remove them before applying in order to have 
# non-linearity
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))


# Step 2 - Max Pooling - Taking the maximum
# Why? Reduce the number of nodes for next Flattening step
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flatten - huge single 1 dimensional vector
classifier.add(Flatten())

# Step 4 - Full Connection
# output_dim/units: don't take too small, don't take too big
# common practice take a power of 2, such as 128, 256, etc.
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units = 1, activation = "sigmoid")) # Checking only one, either apple or banana


# Step 5 - Compiling
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Step 6 - Fitting the CNN to the images
