import tensorflow as tf

output = None
logit_data = [2.0, 1.0, 0.1]
logits = tf.placeholder(tf.float32)

softmax = tf.nn.softmax(logits)

with tf.Session() as session:
    output = session.run(softmax, feed_dict={logits: logit_data})
    print(output)