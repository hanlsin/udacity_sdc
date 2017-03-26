import numpy as np
import math
from data_prep import features, targets, features_test, targets_test


# This code uses a data file, binary.csv.
# This dataset has three input features:
#   GRE score,
#   GPA, and
#   the rank of the undergraduate school (numbered 1 through 4).
# Institutions with rank 1 have the highest prestige, those with rank 4 have the lowest.

# The goal here is to predict if a student will be admitted to a graduate program based on these features.

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Use to same seed to make debugging easier
np.random.seed(42)

# get row size and column size
n_records, n_features = features.shape
last_loss = None

# Initialize weights
#s = math.sqrt(n_features) == n_features**.5
#print(s)
#print(n_features**.5)
weights = np.random.normal(scale=1 / math.sqrt(n_features), size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # TODO: Caluculate the output
        #print(x)
        #print(weights)
        output = sigmoid(np.dot(x, weights))

        # TODO: Calculate the error
        error = y - output

        # TODO: Calculate change in weights
        error_term = error * output * (1-output)
        #print("error_term=", error_term)
        del_w += error_term * x

    # TODO: Update weights
    weights += learnrate * (del_w / n_records)

    # Printing out the mean square error on the training set
    if e % (epochs/10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out-targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, " WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))