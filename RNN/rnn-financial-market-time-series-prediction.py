# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 19:41:25 2018

@author: Mahbub - A - Rob
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read Data
train_df = pd.read_csv("D:\\MahbubProjects\\MachineLearning\\DeepLearning\\Datasets\\RNN\\Google_Stock_Price_Train.csv")

# Take the columns we need which is "open" price column, index 1
# Make it a two dimensional array using 1:2, here 2 is excluded, so will take only 1
train_set = train_df.iloc[:, 1:2].values


# Feature Scaling
# We will use Normalization instead of Standardization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_set = sc.fit_transform(train_set)

"""
# Get input and output
# Input: We have 1258 rows, so the number we need is 0 to 1257
# Input: stock price at time T, 
# Output: Stock price at time T+1
# So output: 1 to 1258
# Look at the generated X_train's opening price and 
# y_train's closing price and compare it with train_set
"""
X_train = train_set[0:1257]
y_train = train_set[1:1258]


