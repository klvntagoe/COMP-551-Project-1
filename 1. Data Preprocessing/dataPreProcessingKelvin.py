import json
import numpy as np


#DATA LOAD
with open("proj1_data.json") as fp:
    data = json.load(fp)


#PRIMARY VARIABLES
X = []
y = []
training_X = []
training_Y = []
cross_validation_X = []
cross_validation_Y = []
testing_X = []
testing_Y = []


#DATA CONSTRUCTION
def construct_dataset(d):
    for p in d:
        temp = []
        for info_name, info_value in p.items():
            if (info_name == "popularity_score"):
                y.append(info_value)
            if (info_name == "is_root"):
                if (info_value == "true"):
                    y.append(1)
                else:
                    y.append(0)
            else:
                temp.append(info_value)
        print(".")
        X.append(temp)


#DATA SEPARATION
def split_dataset(X,y):
    for i in range(10000):
        training_X.append(X[i])
        training_Y.append(y[i])
        print("-")
    for i in range(1000):
        cross_validation_X.append(X[i + 10000])
        cross_validation_Y.append(y[i + 10000])
        print("-")
    for i in range(1000):
        testing_X.append(X[i + 11000])
        testing_Y.append(y[i + 11000])
        print("-")


#FUNCTION CALLS
construct_dataset(data)
print("Dataset Constructed\n")
split_dataset(X,y)
print("Dataset Split\n")


#SAVING DATA
np.save("training_data_X.npy", training_X)
np.save("training_data_Y.npy", training_Y)
np.save("cross_validation_data_X.npy", cross_validation_X)
np.save("cross_validation_data_Y.npy", cross_validation_Y)
np.save("testing_data_X.npy", testing_X)
np.save("testing_data_Y.npy", testing_Y)
print("Dataset Saved\n")