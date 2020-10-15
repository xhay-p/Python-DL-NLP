import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

inp_data = np.linspace(0, 100, 100, dtype=np.float32).reshape(25,4)

#Creating variable with tf.get_variable method
Weights = tf.get_variable("Weigts", shape=(1, 4), initializer=tf.random_uniform_initializer())
Bias = tf.get_variable("Bias", initializer=tf.random.normal([25,1]))

#creating a placeholder
inputs = tf.placeholder(shape=[25,4], dtype=tf.float32)

y = tf.matmul(inputs, tf.transpose(Weights)) + Bias

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs_linear', tf.get_default_graph())
    sess.run(tf.initialize_all_variables())
    res = sess.run(y, feed_dict={inputs:inp_data})
    print(res)