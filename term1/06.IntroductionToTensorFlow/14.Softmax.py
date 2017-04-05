import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

logits = [1.0, 2.0, 3.0]
"""
[ 0.09003057  0.24472847  0.66524096]
"""
print(softmax(logits))

# logits is a two-dimensional array
logits = np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]])
# softmax will return a two-dimensional array with the same shape
"""
[
    [ 0.09003057  0.00242826  0.01587624  0.33333333]
    [ 0.24472847  0.01794253  0.11731043  0.33333333]
    [ 0.66524096  0.97962921  0.86681333  0.33333333]
  ]
"""
print(softmax(logits))