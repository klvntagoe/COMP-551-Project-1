#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#LOAD DATA
weights = []
training_X = np.load("training_data_X.npy")
training_Y = np.load("training_data_Y.npy")
cross_validation_X = np.load("cross_validation_data_X.npy")
cross_validation_Y = np.load("cross_validation_data_Y.npy")
testing_X = np.load("testing_data_X.npy")
testing_Y = np.load("testing_data_Y.npy")
print("Dataset Loaded\n")


# In[3]:


#NORMAL EQUATION SOLVER
def normalEquation(X, Y):
    XT = X.transpose()
    XTX = np.matmul(XT, X)
    a = np.linalg.solve(XTX, np.identity(len(XTX)))
    temp = np.matmul(a, XT)
    w = np.matmul(temp, Y)
    return w


# In[4]:


#WEIGHTS LEARNING FUNCTION CALLS
weights = normalEquation(training_X, training_Y)
print("Weights Learned\n")


# In[5]:


#SAVING WEIGHTS
np.save("weights_NE.npy", weights)
print("Weights Saved\n")

