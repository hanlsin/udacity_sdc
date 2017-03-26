from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

for f in (np.float32, np.float64, float):
    finfo = np.finfo(f)
    print(finfo.dtype, finfo.nexp, finfo.nmant)

n_input = 28 * 28 # image shape
n_classes = 10 # MNIST total classes

# Import MNIST data
mnist = input_data.read_data_sets('../datasets', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
print(train_features.itemsize)
print(train_features.shape)
print(train_features.itemsize * train_features.shape[0] * train_features.shape[1])
test_features = mnist.test.images
print(test_features.itemsize)
print(test_features.shape)
print(test_features.itemsize * test_features.shape[0] * test_features.shape[1])

train_labels = mnist.train.labels.astype(np.float32)
print(train_labels.itemsize * train_labels.shape[0] * train_labels.shape[1])
test_labels = mnist.test.labels.astype(np.float32)
print(test_labels.itemsize * test_labels.shape[0] * test_labels.shape[1])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    output = sess.run(weights)
    print(output.itemsize * output.shape[0] * output.shape[1])
    output = sess.run(bias)
    print(output.itemsize * output.shape[0])
