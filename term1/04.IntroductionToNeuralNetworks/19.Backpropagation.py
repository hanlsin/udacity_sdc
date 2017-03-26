import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(x))

X = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weight_input_to_hidden = np.array([[0.5, -0.6],
                                   [0.1, -0.2],
                                   [0.1, 0.7]])
weight_hidden_to_output = np.array([0.1, -0.3])

# Forward pass
hidden_layer_input = np.dot(X, weight_input_to_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
print(hidden_layer_output)

output_layer_input = np.dot(hidden_layer_output, weight_hidden_to_output)
output_layer_output = sigmoid(output_layer_input)
print(output_layer_output)

# Backwards pass
# TODO: Calculate error
error = (target - output_layer_output)
print(error)

# TODO: Calculate error gradient for output layer
del_err_output = error * output_layer_output * (1 - output_layer_output)
print(del_err_output)
print("--------------")

# TODO: Calculate error gradient for hidden layer
del_err_hidden = np.dot(del_err_output, weight_hidden_to_output) * \
                 hidden_layer_output * (1 - hidden_layer_output)
print(np.dot(del_err_output, weight_input_to_hidden))
print(hidden_layer_output)
print(del_err_hidden)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * del_err_output * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * del_err_hidden * X[:, None]


print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)