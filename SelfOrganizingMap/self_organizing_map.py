# -*- coding: utf-8 -*-
# Project: Self Organizing Map
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
frauds = np.concatenate((map_dictionary[(8,3)], map_dictionary[(5,7)]), axis=0)
frauds = sc.inverse_transform(frauds) # These are the customer ids = possible fraud
