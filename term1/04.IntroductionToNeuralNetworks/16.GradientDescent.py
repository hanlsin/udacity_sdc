import numpy as np


# Defining the sigmoid function for activations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Input data
x = np.array([1, 2])
# Target
y = 0.5
# Input to output weights
weights = np.array([0.5, -0.5])

# The learning rate, eta in the weight step equation
learnrate = 0.5

# The neural network output (y-hat)
# nn_output = sigmoid(x[0] * weights[0] + x[1] * weights[1])
# or
nn_output = sigmoid(np.dot(x, weights))

# output error (y - y-hat)
error = y - nn_output

# error term (lowercase delta)
error_term = error * sigmoid_prime(np.dot(x, weights))
print(error_term)
error_term = error * nn_output * (1 - nn_output)
print(error_term)

# Gradient descent step
del_w = [learnrate * error_term * x[0], learnrate * error_term * x[1]]

print('Neural Network Output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
