# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 18:57:26 2018

@author: Mahbub
"""

# =============================================================================
# 
# Boltzmann machine
# A Boltzmann machine (also called stochastic Hopfield network with hidden units)
# is a type of stochastic recurrent neural network (and Markov random field). 
# The Boltzmann machine is a Monte Carlo version of the Hopfield network. 

# They were one of the first neural networks capable of learning internal 
# representations, and are able to represent and (given sufficient time) solve 
# difficult combinatoric problems. 
# 
# The Boltzmann machine would theoretically be a rather general computational 
# medium. For instance, if trained on photographs, the machine would theoretically 
# model the distribution of photographs, and could use that model to, for example, 
# complete a partial photograph.
# 
# Restricted Boltzmann machine RBM
# "RBM" which does not allow intralayer connections between hidden units. After 
# training one RBM, the activities of its hidden units can be treated as data 
# for training a higher-level RBM. This method of stacking RBMs makes it possible 
# to train many layers of hidden units efficiently and is one of the most common 
# deep learning strategies. 

# Deep Boltzmann machine DBM
# A deep Boltzmann machine (DBM) is a type of binary pairwise Markov random 
# field (undirected probabilistic graphical model) with multiple layers of 
# hidden random variables.

# Like DBNs, DBMs can learn complex and abstract internal representations of 
# the input in tasks such as object or speech recognition, using limited, 
# labeled data to fine-tune the representations built using a large supply of 
# unlabeled sensory input data. However, unlike DBNs and deep convolutional 
# neural networks, they adopt the inference and training procedure in both 
# directions, bottom-up and top-down pass, which allow the Deep Boltzmann 
# machine to better unveil the representations of the input structures.

# Deep belief network DBN
# In machine learning, a deep belief network (DBN) is a generative graphical 
# model, or alternatively a class of deep neural network, composed of multiple 
# layers of latent variables ("hidden units"), with connections between the 
# layers but not between units within each layer.

# =============================================================================

"""
Supervised: Artificial Neural Network - Regression, Classification
Supervised: Convolution Neural Network - Computer Vision
Supervised: Recurrent Neural Network - Time Series

Unsupervised: Self Organizing Map - Feature Detection
Unsupervised: Deep Boltzmann Machine - Recommender System
Unsupervised: Auto Encoder - Recommender System             
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable



# Read dataset
movies_df = pd.read_csv("ml-1m/movies.dat", sep = "::", header = None, 
                        engine = "python", encoding = "latin-1")
movies_df.head()
movies_df.info()
movies_df.describe().transpose()

users_df = pd.read_csv("ml-1m/users.dat", sep = "::", header = None, 
                        engine = "python", encoding = "latin-1")
users_df.head()
users_df.info()
users_df.describe().transpose()


# 0 = User Id
# 1 = Movie Id
# 2 = Rating
# 3 = Time stamps of rating given
ratings_df = pd.read_csv("ml-1m/ratings.dat", sep = "::", header = None, 
                        engine = "python", encoding = "latin-1")
ratings_df.head()



# Prepare training and test set
# Remember, as in ml-100k, there are five train and test sets, we should apply
# five fold cross validation
# Be sure to check the data seperator, it might be :: or tab or 
# 0 = User id
# 1 = Movie Id
# 2 = Rating
# 3 = Time Stamp
train_df = pd.read_csv("ml-100k/u1.base", delimiter = "\t")
train_df.head()

train_df.info()
train_df.describe().transpose()

train_array = np.array(train_df, dtype = "int")
print(train_array)



test_df = pd.read_csv("ml-100k/u1.test", delimiter = "\t")
test_df.head()

test_df.info()
test_df.describe().transpose()

test_array = np.array(test_df, dtype = "int")
print(test_array)


# =============================================================================
# https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
# 2 and 1 dimensional Slicing Note

# data = np.array([[11, 12, 13],
# 		      [43, 53, 63],
# 		      [73, 83, 93]])
# # separate data
# X = data[:, :-1] # 2 dimensional
# y = data[:, -1]  # 1 dimensional
# print("2 Dimension X[:, :-1] =\n", X)
# print("1 Dimension y[:, -1] =\n", y)

# 2 Dimension X[:, :-1] =
#  [[11 12]
#  [43 53]
#  [73 83]]
# 1 Dimension y[:, -1] =
#  [13 63 93]
# =============================================================================


# =============================================================================
# data = np.array([11, 22, 33, 44, 55])
# print(data[-5:])
# print(data[0:1])
# print(data[2:5])
# print(data[-2:])
# print(data[-3:])
# 
# [11 22 33 44 55]
# [11]
# [33 44 55]
# [44 55]
# [33 44 55]
# =============================================================================


# =============================================================================
# data = np.array([[11, 12, 13],
#  		      [43, 53, 63],
#  		      [73, 83, 93]])
# print(data[:, 0])
# print(max(data[:, 0]))
# =============================================================================


# get number of users and number of movies
# Slicing
# 0 = first column = users
# 1 = second column = movies
print(train_array[:, 0]) 
print(max(train_array[:, 0]))
# [  1   1   1 ... 943 943 943]
# 943

print(test_array[:, 0])
print(max(test_array[:, 0]))
# [  1   1   1 ... 459 460 462]
# 462

# conver to integer
print(int(max(test_array[:, 0])))


# total number of users
total_users = int(max(max(train_array[:, 0]), max(test_array[:, 0])))
print(total_users)


total_movies = int(max(max(train_array[:, 1]), max(test_array[:, 1])))
print(total_movies)


# Create a list where
# user id is first column, movie id is 2nd, all ratings 1682 are other columns

# For torch, we need list of lists
# Meaning one list for each user, so 943 lists for 943 users
# each list contains ratings for 1682 movies

# Algorithm to make a 2 dimensional array
# take the data as input
# Take an id from all the ids (total_users+1)
# loop through the movies
# save movie ratings for one id
# start again loop
# data = train array
# data[:, 0] = all rows, 0th column (First column)
# data[:, 1] = all rows, 1th(2nd) column 
# range(1, 5) : take 1 to 4
def convert(data):
    new_data = []
    for user_id in range(1, total_users + 1):
        # take whole movie id column from train array
        # save these movie ids to new_data where data.user_id == user.user_id
        movie_id = data[:, 1][user_id == data[:, 0]]
        
        # take all the ratings where data.user_id == user.user_id
        # save all the ratings in rating_id
        rating_id = data[:, 2][user_id == data[:, 0]]
        
        # initialize all the ratings as zero
        # fill the zeroes with original ratings if any
        # how many zeroes? == total_movies
        ratings = np.zeros(total_movies)
        
        # save all the rating_id in ratings list corresponding movie id
        # movie_id started with 1
        # if you need to map non-integer movie id
        # you can do movie_id[].....
        ratings[movie_id - 1] = rating_id
        
        # save all the ratings and build a list of lists
        new_data.append(list(ratings))
        
    return new_data

# Creating the list    
train_list = convert(train_array)
test_list = convert(test_array)
        

# Converting into pytorch tensors
train_tensor = torch.Float








