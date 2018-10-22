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



"""
# Reshaping : Add extra dimension
# We reshape: We have 2D array where 1st dimension has 1257 rows/observations
# And 2nd dimension has 1 feature which is stock price at time T
# Our input is T and output is T+1
# So, T + 1 - T = 1
# So the time step is 1

"""

# Use keras RNN doc
# (1257, 1, 1) = first dimension, time step, feature
X_train = np.reshape(X_train, (1257, 1, 1))


# Import keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# Initializing the RNN
regressor = Sequential()


# Add input layer
# units = number of memory units, using 4 is a common practice
# Input shape none so that the model can accept any time step
# input shape = None, 1, 1 is for number of features we have
regressor.add(LSTM(units = 4, activation = "sigmoid", input_shape = (None, 1)))

# Add output layer
# units = 1 as output has one dimension
regressor.add(Dense(units = 1))


# Compiling RNN
regressor.compile(optimizer="adam", loss="mean_squared_error")

# Fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size=32, epochs = 200)

# Make a prediction

# Get the real stock price of 2017
test_df = pd.read_csv("D:\\MahbubProjects\\MachineLearning\\DeepLearning\\Datasets\\RNN\\Google_Stock_Price_Test.csv")
real_stock_price = test_df.iloc[:, 1:2].values

# Predict stock price of 2017
inputs = real_stock_price
inputs = sc.transform(inputs) # As we scaled the train set
inputs = np.reshape(inputs, (20, 1, 1)) # We need to reshape as our train set is reshaped
predicted_stock_price = regressor.predict(inputs) # This is a scaled version
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualize the prediction
plt.plot(real_stock_price, color="green", label="Real Google Stock Prices")
plt.plot(predicted_stock_price, color="black", label="Predicted Google Stock Prices")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()


# Get the real stock price from 2012 to 2016
real_stock_price_train_df = pd.read_csv("D:\\MahbubProjects\\MachineLearning\\DeepLearning\\Datasets\\RNN\\Google_Stock_Price_Train.csv")
real_stock_price_train = real_stock_price_train_df.iloc[:, 1:2].values

# Get the predicted stock price from 2012 to 2016
predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)

# Visualize the result
plt.plot(real_stock_price_train, color="red", label="Real Google Stock Prices")
plt.plot(predicted_stock_price_train, color="black", label="Predicted Google Stock Prices")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
