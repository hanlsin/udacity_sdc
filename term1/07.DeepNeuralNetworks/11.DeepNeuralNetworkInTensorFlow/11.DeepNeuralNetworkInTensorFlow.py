import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load examples
# You can find this and many more examples of TensorFlow at
# https://github.com/aymericdamien/TensorFlow-Examples
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

# Learning parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128  # Decrease batch size if you don't have enough memory
display_step = 1

n_input = 784   # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Hidden layer parameters
n_hidden_layer = 256    # layer number of features

# Weights and Biases
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])

# Multilayer perception
## hidden layer
hidden_layer = tf.add(tf.matmul(x_flat, weights['hidden_layer']),
                      biases['hidden_layer'])
## hidden layer ReLU
hidden_layer = tf.nn.relu(hidden_layer)
## output layer
logits = tf.add(tf.matmul(hidden_layer, weights['out']), biases['out'])

# Optimizer
## loss or cost = D(L, S)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=y))
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cost)

# Session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)

        # loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # run optimization
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        # display logs per epoch step
        if epoch % display_step == 0:
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                  "{:.9f}".format(c))
    print("Optimization Finished!")

    # test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # decrease test_size if you don't have enough memory
    test_size = 256
    print("Accuracy:", accuracy.eval({x: mnist.test.images[:test_size],
                                      y: mnist.test.labels[:test_size]}))
