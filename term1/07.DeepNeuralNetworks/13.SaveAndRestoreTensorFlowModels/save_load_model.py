import tensorflow as tf

##################################################
# Saving variables

# Training a model can take hours. But once you close your TensorFlow session,
# you lose all the trained weights and biases. If you were to reuse the model
# in the future, you would have to train it all over again!

save_file = './model.ckpt'

# weights and biases
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# class used to save and/or restore tensor variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # initialize all variables
    sess.run(tf.global_variables_initializer())

    # show the values of weights and biases
    print("Weights:", sess.run(weights))
    print("Bias:", sess.run(bias))

    # save the model
    saver.save(sess, save_file)

##################################################
# Loading variables

# remove the previous weights and bias
tf.reset_default_graph()

# weights and biases
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# class used to save and/or restore tensor variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # initialize all variables
    sess.run(tf.global_variables_initializer())

    # show the values of weights and biases
    print("new Weights:", sess.run(weights))
    print("new Bias:", sess.run(bias))

    # load the model
    saver.restore(sess, save_file)

    print("Loaded Weights:", sess.run(weights))
    print("Loaded Bias:", sess.run(bias))

##################################################
# Save a Trained Model
print("Save a Trained Model")

# remove the previous weights and bias
tf.reset_default_graph()

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Learning parameters
learning_rate = 0.001
n_features = 784
n_classes = 10
batch_size = 128
n_epochs = 500

mnist = input_data.read_data_sets('.', one_hot=True)

# Features and labels
features = tf.placeholder(tf.float32, [None, n_features])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights and biases
weights = tf.Variable(tf.random_normal([n_features, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Output layer
logits = tf.add(tf.matmul(features, weights), bias)

# Loss or cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels))
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Save file
save_file = './train_model.ckpt'

# class used to save and/or restore tensor variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # initialize all variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(n_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)

        # loop over all batches
        for i in range(batch_size):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={features: batch_features,
                                           labels: batch_labels})

        # display logs per 10 epoch step
        if epoch % 100 == 0:
            c = sess.run(cost, feed_dict={features: batch_features,
                                          labels: batch_labels})
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                  "{:.9f}".format(c))
            valid_accuracy = sess.run(
                accuracy,
                feed_dict={
                    features: mnist.validation.images,
                    labels: mnist.validation.labels
                })
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))

    print("Optimization Finished!")

    # load the model
    saver.save(sess, save_file)
    print("Trained model saved.")

##################################################
# Load a Trained Model
print("Load a Trained Model")

save = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
