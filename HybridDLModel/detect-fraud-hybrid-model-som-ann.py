# -*- coding: utf-8 -*-
# Project: Hybrid Deep Learning Model
"""
http://davis.wpi.edu/~matt/courses/soms/
Self-organizing maps (SOMs) are a data visualization technique invented by Professor 
Teuvo Kohonen which reduce the dimensions of data through the use of self-organizing 
neural networks. The problem that data visualization attempts to solve  is that humans 
simply cannot visualize high dimensional data as is so techniques are created to help us 
understand this high dimensional data. Two other techniques of reducing the dimensions 
of data that has been presented in this course has been N-Land and Multi-dimensional 
Scaling. The way  SOMs go about reducing dimensions is by producing a map of usually 
1 or 2 dimensions which plot the similarities of the data by grouping similar data 
items together.  So SOMs accomplish two things, they reduce dimensions and display similarities. 
"""
"""
Created on Tue Oct 23 19:05:13 2018

@author: Mahbub - A - Rob
"""

# Import Librabires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read Data
# Dataset: UCI Machine Learning Repository http://archive.ics.uci.edu/ml/datasets/Statlog+Project
# Other Datasets: http://archive.ics.uci.edu/ml/index.php
# 0 means application of the customer is not approved
credit_card_applications_df = pd.read_csv("Credit_Card_Applications.csv")

# Take all the input columns except the last one
X = credit_card_applications_df.iloc[:, :-1].values

# Take only the last column
y = credit_card_applications_df.iloc[:, -1].values

# Feature scaling: Use Normalization MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)


# We will be suing minisom library here
from minisom import MiniSom

# Out dataset is small
# So we will just create a 10 by 10 matrix, X = 10, y = 10
# Input_len = number of features in our dataset 14+1=15
# Sigma: Radious in the different neighborhoods in the grid
minisom = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
minisom.random_weights_init(X)
minisom.train_random(data = X, num_iteration = 200)

# Visualize the SOM
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(minisom.distance_map().T)
colorbar() # Color close to white are frauds
markers = ["o", "s"] # s=square
colors = ["r", "g"] # red = didn't get approval, green=got approval
for i, x in enumerate(X):
    winning_node = minisom.winner(x)
    plot(winning_node[0] + 0.5, # placing to center using .5 
         winning_node[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = "None",
         markersize = 10,
         markeredgewidth = 2
         )
show()

# Detect Fraud
# Create a dictionary and map all the winning nodes to customer
# Key will represent the co-ordinates of the winning nodes
# For spyder click on the numpy array cell, a new window will be opened
# In the new window, each line will corresponds to one customer that
# associated with the winning node of that matching co-ordinates
# Click again and we will see the value of the customer ID but it is scaled
# We will have to inverse the scaling
# exis = 0 means concatenate vertically which is default
map_dictionary = minisom.win_map(X) # Pass the whole dataset
frauds = np.concatenate((map_dictionary[(2,7)], map_dictionary[(4,9)]), axis=0)
frauds = sc.inverse_transform(frauds) # These are the customer ids = possible fraud

# Create hybrid model
# Going from Unsupervised to Supervised Deep Learning
# Create matrix of features: customers
# Include all the customers
customers = credit_card_applications_df.iloc[:, 1:].values # take all except the last column

is_fraud = np.zeros(len(credit_card_applications_df))
for i in range(len(credit_card_applications_df)):
    # if customer id 
    if credit_card_applications_df.iloc[i, 0] in frauds: # 0 = customer id column
        is_fraud[i] = 1


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Apply ANN
#import keras
from keras.models import Sequential
from keras.layers import Dense

# Improve ANN
from keras.layers import Dropout # To avoid overfitting, underfitting
# Initialising ANN
classifier = Sequential()


# Apply ANN
classifier.add(Dense(activation="relu", input_dim=15, units=2, kernel_initializer="uniform"))
classifier.add(Dropout(p = 0.1)) # Use 0.1 to .4

# Add output layer
# Only one output so, units = 1
# Activation Function = sigmoid
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    
# Compiling the ANN
# Optimizer = the algorithm you want to use
# loss function within stachastic gradient
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

# Fitting ANN to the training set
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Predicting the Test set results
y_pred = classifier.predict(customers)

# Add our prediction array to the data frame
# Take only the first column
# As y-red is 2 dimensional, so we need to take the range as 2 dimensional hence 0:1
# And we need horizontal concatenation hence axis = 1
y_pred = np.concatenate((credit_card_applications_df.iloc[:, 0:1], y_pred), axis = 1)

# Sort the predicted column
# We want to sort second column which index is 1
y_pred = y_pred[y_pred[:, 1].argsort()]


