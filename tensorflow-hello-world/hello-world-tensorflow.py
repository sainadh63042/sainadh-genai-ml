import tensorflow as tf

hello = tf.constant("hello tensorflow")
print(hello)
print(hello.numpy().decode('utf-8'))
