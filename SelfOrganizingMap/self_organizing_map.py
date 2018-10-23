# -*- coding: utf-8 -*-
# Project: Self Organizing Map

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
minisom.train_random(data = X, num_iteration = 100)

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