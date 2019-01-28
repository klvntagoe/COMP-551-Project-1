#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
from collections import Counter


# In[2]:


#DATA LOADING
with open("Project 1 Materials\proj1_data.json") as fp:
    data = json.load(fp)


# In[3]:


#PRIMARY VARIABLES
labels = data[0].keys()
mostFrequentWords = {}
rankingOfMostFrequentWords = {}
X = []
y = []
training_X = []
training_Y = []
cross_validation_X = []
cross_validation_Y = []
testing_X = []
testing_Y = []


# In[4]:


#TEXT PREPROCESSING

#Splits a given text into individual tokens
def splitText(text):
    lowerCaseText =  text.lower()
    splitText = lowerCaseText.split()
    return splitText

#Find top 160 frequently occurring word in 'data'
def topFrequencies(dictionaryList):
    countVariable = Counter()
    for item in dictionaryList:
        splitted_text = splitText(item['text'])
        for word in splitted_text:
            countVariable[word] += 1
    return dict(countVariable.most_common(160))

#Convert list of tuples of (word, frequency) to (word, ranking)
def convertTupleList(list):
    i = 0
    newList = {}
    for word in list:
        newList[word] = i
        i+=1
    return newList

#Construct a word count vector from a comment
def constructWordCountVector(text):
    vector = [0]*160
    splitted_text = splitText(text)
    for word in splitted_text:
        i = rankingOfMostFrequentWords.get(word, -1)
        if (i != -1): vector[i] += 1
    return vector

mostFrequentWords = topFrequencies(data)
rankingOfMostFrequentWords = convertTupleList(mostFrequentWords)


# In[5]:


#DATASET CONSTRUCTION
def construct_dataset(d):
    for p in d:
        training_example = []
        for key, value in p.items():
            if (key == "popularity_score"):
                y.append(value)
            elif (key == "is_root"):
                if (value == True): training_example.append(1)
                elif (value == False): training_example.append(0)
            elif (key == "text"):
                vector = constructWordCountVector(value)
                training_example.extend(vector)
            else:
                training_example.append(value)
        X.append(training_example)


# In[6]:


#DATASET SEPARATION
def split_dataset(X,y):
    for i in range(10000):
        training_X.append(X[i])
        training_Y.append(y[i])
    for i in range(1000):
        cross_validation_X.append(X[i + 10000])
        cross_validation_Y.append(y[i + 10000])
    for i in range(1000):
        testing_X.append(X[i + 11000])
        testing_Y.append(y[i + 11000])


# In[7]:


#DATASET GENERATION FUNCTION CALLS
construct_dataset(data)
print("Dataset Constructed\n")
split_dataset(X,y)
print("Dataset Split\n")


# In[8]:


#SAVING DATASET
np.save("training_data_X.npy", training_X)
np.save("training_data_Y.npy", training_Y)
np.save("cross_validation_data_X.npy", cross_validation_X)
np.save("cross_validation_data_Y.npy", cross_validation_Y)
np.save("testing_data_X.npy", testing_X)
np.save("testing_data_Y.npy", testing_Y)
print("Dataset Saved\n")


# In[9]:


#TRAINING EXAMPLE TEST RUN
import random
i = random.randint(1,12000)
print(data[i])
print(X[i])
print(y[i])


# In[10]:


#COUNTING NUMBER OF IMPROPER TRAINING EXAMPLES
count = 0
i = 0
for line in X:
    if len(line) < 163: count+=1
print(count)


# In[11]:


#COUNTING PROPORTIONS OF BOOLEAN FEATURES
numRTrue = 0   #Count for isRoot = True
numRFalse = 0  #Count for isRoot = False
numCTrue = 0   #Count for Controversiality = True
numCFalse = 0  #Count for Controversiality = False
i = 0
for line in X:
    if (line[160] == 1): numRTrue+=1
    else: numRFalse += 1
    if (line[161] == 1): numCTrue+=1
    else: numCFalse += 1
    i+=1
print(numRTrue)
print(numRFalse)
print(numRTrue + numRFalse)
print(numCTrue)
print(numCFalse)
print(numCTrue + numCFalse)

