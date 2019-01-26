import json
import numpy as np
from collections import Counter



#DATA LOAD
with open("proj1_data.json") as fp:
    data = json.load(fp)

'''
Label Ordering
    {'text': String ,
        'is_root': boolean,
        'controversiality': int,
        'children': float,
        'popularity_score': float
    }
'''

#PRIMARY VARIABLES
countVariable = Counter()
labels = data[0].keys()
mostFrequentWords = {}
X = []
y = []
training_X = []
training_Y = []
cross_validation_X = []
cross_validation_Y = []
testing_X = []
testing_Y = []
counter = Counter()



#TEXT PREPROCESSING

#Splits a given text into individual tokens
def splitText(text):
    lowerCaseText =  text.lower()
    splitText = lowerCaseText.split()
    return splitText

#Count the frequencies of tokens
def countFrequencies(textArray):
    for word in textArray:
        countVariable[word] += 1

#Find top 160 frequently occurring word in 'data'
def topFrequencies(dictionaryList):
    for item in dictionaryList:
        splited_text = splitText(item['text'])
        countFrequencies(splited_text)
        return dict(countVariable.most_common(160))

mostFrequentWords = topFrequencies(data)



#DATASET CONSTRUCTION
def construct_dataset(d):
    for p in d:
        temp = []
        for name, value in p.items():
            if (name == "popularity_score"):
                y.append(value)
            elif (name == "is_root"):
                if (value == True): temp.append(1)
                elif (value == False): temp.append(0)
            elif (name == "text"):
                print(".")
                #WHOLE LOTTA (GANG) SHIT TO BE DONE HERE
            else:
                temp.append(value)
        X.append(temp)



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



#DATASET GENERATION FUNCTION CALLS
construct_dataset(data)
print("Dataset Constructed\n")
split_dataset(X,y)
print("Dataset Split\n")



#SAVING DATASET
np.save("training_data_X.npy", training_X)
np.save("training_data_Y.npy", training_Y)
np.save("cross_validation_data_X.npy", cross_validation_X)
np.save("cross_validation_data_Y.npy", cross_validation_Y)
np.save("testing_data_X.npy", testing_X)
np.save("testing_data_Y.npy", testing_Y)
print("Dataset Saved\n")