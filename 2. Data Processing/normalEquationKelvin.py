import numpy as np 


#LOADING DATA
weights = []
training_X = np.load("training_data_X.npy")
training_Y = np.load("training_data_Y.npy")
cross_validation_X = np.load("cross_validation_data_X.npy")
cross_validation_Y = np.load("cross_validation_data_Y.npy")
testing_X = np.load("testing_data_X.npy")
testing_Y = np.load("testing_data_Y.npy")
print("Dataset Loaded\n")

