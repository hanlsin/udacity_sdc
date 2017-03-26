import tensorflow as tf

def run(t):
        with tf.Session() as sess:
                output = sess.run(t)

        return output

print(run(tf.constant(10)))

x = tf.add(tf.constant(5), tf.constant(2))
print(run(x))

y = tf.subtract(10, 4)
print(run(y))

z = tf.multiply(2, 5)
print(run(z))

# Error
#tf.subtract(tf.constant(2.0), tf.constant(1))
# have to cast
r = tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))
print(run(r))

print("-------------")
print(run(tf.divide(tf.constant(10), tf.constant(2))))
print(run(tf.subtract(tf.divide(tf.constant(10), tf.constant(2)), tf.cast(tf.constant(1), tf.float64))))