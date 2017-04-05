import tensorflow as tf

x = tf.Variable(5)

# initialize all TensorFlow variables from the graph.
# You call the operation using a session to initialize all the variables as shown above. 
init = tf.global_variables_initializer()

n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
bias = tf.Variable(tf.zeros(n_labels))

with tf.Session() as sess:
        sess.run(init)
        print(weights.get_shape())
        print(bias.get_shape())

# Linear Classifier Quiz
"""
You'll be classifying the handwritten numbers 0, 1, and 2 
from the MNIST dataset using TensorFlow. 
The above is a small sample of the data you'll be training on. 
Notice how some of the 1s are written with a serif at the top 
and at different angles. The similarities and differences 
will play a part in shaping the weights of the model.
"""