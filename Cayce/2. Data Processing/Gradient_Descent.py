import numpy as np
import math
def gradient_descent(X, y, initial_weights, beta, eta, epsilon) :

    estimated_weights = []
    new_weights = initial_weights
    i = 1
    XT = X.transpose()
    XTX = np.matmul(XT, X)
    XTy = np.matmul(XT, y)

    while True :

        old_weights = new_weights
        #need to choose how to increment beta, line below will be placeholder
        beta = beta + 1
        alpha = eta/(1 + beta)
        new_weights  = old_weights - (2*alpha)*((np.matmul(old_weights,XTX) - XTy))

        accumulator = 0
        #for i in range (len(initial_weights) - 1) :
        #    new_weights_i = new_weights[i]
        #    old_weights_i = old_weights[i]
        #    accumulator = accumulator + math.pow(new_weights_i - old_weights_i,2)
        norm = np.linalg.norm(new_weights[i] - old_weights[i])
        if norm <= epsilon :
            estimated_weights = new_weights
            break
        else:
            i = i + 1
    return estimated_weights


X = np.load("/Users/CayceM/Documents/GitHub/COMP-551-Project-1/Cayce/2. Data Processing/testing_data_X_1.npy")
Y = np.load("/Users/CayceM/Documents/GitHub/COMP-551-Project-1/Cayce/2. Data Processing/testing_data_Y_1.npy")

initial_weights = [0]*164
weights = gradient_descent(X, Y, initial_weights, 1, 1, 1)
