import numpy as np
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
        for i in range (initial_weights.length() - 1) :
            accumulator = accumulator + math.pow(new_weights[i] - old_weights[i],2)
        norm =  math.sqrt(accumulator)
        if norm <= epsilon :
            estimated_weights = new_weights
            break
        else:
            i = i + 1
    return estimated_weights
X = np.load("testing_data_X_1.npy")
Y = np.load("testing_data_Y_1.npy")
initial_weights = [0]*165
weights = gradient_descent(X, Y, initial_weights, 1, 1, 1)
