import tensorflow as tf

# In TensorFlow, data isn’t stored as integers, floats, or strings. 
# These values are encapsulated in an object called a tensor. 
# In the case of hello_constant = tf.constant('Hello World!'), 
# hello_constant is a 0-dimensional string tensor, 
# but tensors come in a variety of sizes as shown below:
hello_constant = tf.constant('Hello World')
# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
 # C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])

print("TensorFlow")

with tf.Session() as sess:
    output = sess.run(hello_constant)
    print(output)