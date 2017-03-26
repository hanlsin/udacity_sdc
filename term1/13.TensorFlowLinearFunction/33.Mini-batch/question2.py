import tensorflow as tf
import numpy as np

n_input = 28 * 28   # image shape
n_classes = 10      # MNIST total classes

# features and labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

"""
Question 2
Use the parameters below, how many batches are there, and what is the last batch size?

features is (50000, 400)

labels is (50000, 10)

batch_size is 128
"""

print(50000 / 128)
print(50000 % 128)
