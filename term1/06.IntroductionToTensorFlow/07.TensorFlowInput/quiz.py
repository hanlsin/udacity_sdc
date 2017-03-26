# Solution is available in the other "solution.py" tab
import tensorflow as tf

x = tf.placeholder(tf.string)
with tf.Session() as sess:
        output = sess.run(x, feed_dict={x: 'Hello World'})
        print(output)

y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)
with tf.Session() as sess:
        output = sess.run([x, y, z], feed_dict={x: 'Hello World', y: 123, z: 456.789})
        print(output)

def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        # TODO: Feed the x tensor 123
        output = sess.run(x, feed_dict={x: 123})

    return output

print(run())