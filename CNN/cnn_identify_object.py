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
from keras.preprocessing.image import ImageDataGenerator

# code from keras doc https://keras.io/preprocessing/image/
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset\\training_set',
                                                    target_size=(64, 64), # keep same as hidden layers
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset\\test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000, # We have 8000 images training set
                        epochs=2, # incrase this number on more powerful cpu
                        validation_data=test_set,
                        validation_steps=2000) # We have 2000 images in test set




# Second Part

# Detect single object - Predict single object
import numpy as np
from keras.preprocessing import image


# Load image
test_image = image.load_img("dataset\\single_prediction\\cat_or_dog_2.jpg", 
                            target_size = (64, 64))

# Let's make it a 3D array as input is a 3D; input_shape=(64, 64, 3)
test_image = image.img_to_array(test_image) # Returns 3D array

# We need 4D array to predict, so add another dimension in the first position, axis = 0
test_image = np.expand_dims(test_image, axis = 0)

# Predict - it expects 4D array, above code place 1 at the first position
result = classifier.predict(test_image) # returns 0 or 1

# Check the value of 0 and 1 in the training set
training_set.class_indices

# Check first row and first column value
if result[0][0] == 1:
    prediction = "Dog"
else:
    prediction = "Cat"








