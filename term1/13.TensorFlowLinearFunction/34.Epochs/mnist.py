from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def batches(batch_size, features, labels):
    out_batches = []
    size = len(features)
    for i in  range(0, size, batch_size):
        end = i + batch_size
        batch = [features[i:end], labels[i:end]]
        out_batches.append(batch)

    return out_batches

def print_epoch_stats(epoch_i, sess, last_features, last_labels):
    """
    Print cost and validation accuracy of an epoch
    """
    current_cost = sess.run(
        cost,
        feed_dict={features: last_features, labels: last_labels})
    valid_accuracy = sess.run(
        accuracy,
        feed_dict={features: valid_features, labels: valid_labels})
    print('Epoch: {:<4} - Cost: {:<8.3} Valid Accuracy: {:<5.3}'.format(
        epoch_i,
        current_cost,
        valid_accuracy))

# Import MNIST data
mnist = input_data.read_data_sets('../datasets', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
print(train_features.shape)
valid_features = mnist.validation.images
print(valid_features.shape)
test_features = mnist.test.images
print(test_features.shape)

n_input = train_features.shape[1]  # MNIST data input (img shape: 28*28)

train_labels = mnist.train.labels.astype(np.float32)
print(train_labels.shape)
valid_labels = mnist.validation.labels.astype(np.float32)
print(valid_labels.shape)
test_labels = mnist.test.labels.astype(np.float32)
print(test_labels.shape)

n_classes = train_labels.shape[1]  # MNIST total classes (0-9 digits)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits y = xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

batch_size = 128
epochs = 100
learn_rate = 0.1

train_batches = batches(batch_size, train_features, train_labels)

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch_i in range(epochs):

        # Loop over all batches
        for batch_features, batch_labels in train_batches:
            train_feed_dict = {
                features: batch_features,
                labels: batch_labels,
                learning_rate: learn_rate}
            sess.run(optimizer, feed_dict=train_feed_dict)

        # Print cost and validation accuracy of an epoch
        print_epoch_stats(epoch_i, sess, batch_features, batch_labels)

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})

print('Test Accuracy: {}'.format(test_accuracy))