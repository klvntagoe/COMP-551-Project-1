import numpy as np
import math
def gradient_descent(X, y, initial_weights, beta, eta, epsilon) :

    estimated_weights = []
    new_weights = initial_weights
    i = 1
    XT = X.transpose()
    XTX = np.matmul(XT, X)
    XTy = np.matmul(XT, y)


    norm = 100

    while ( norm > epsilon ) :


        old_weights = new_weights

        #need to choose how to increment beta, line below will be placeholder

        alpha = eta/(1 + beta)
        beta = beta + .5

        matmul = np.matmul(XTX,old_weights)
        subtracted = np.subtract(matmul,XTy)
        subtracted = (2*alpha)*subtracted
        new_weights  = np.subtract(old_weights, subtracted)

        normsubtracted = np.subtract(new_weights, old_weights)

        norm = np.linalg.norm(normsubtracted)

        i = i + 1
        #print(norm)
    estimated_weights = new_weights
    return estimated_weights


X = np.load("/Users/CayceM/Documents/GitHub/COMP-551-Project-1/Cayce/2. Data Processing/training_data_X_1.npy")
y = np.load("/Users/CayceM/Documents/GitHub/COMP-551-Project-1/Cayce/2. Data Processing/training_data_Y_1.npy")

initial_weights = [0]*164
weights = gradient_descent(X, y, initial_weights, 0, .0001, .000001)
