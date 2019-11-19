import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

x = tf.constant(5, name="x")
y = tf.constant(4, name="y")

add = tf.add(x, y, name="add")
mul = tf.multiply(x, y, name="mul")

with tf.Session() as sess:
    #Creates the summary writer
    #After graph definition
    #Before Session
    #Since we not created a graph explicitly,
    #Every operation is being done on default_graph
    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    a, m = sess.run(fetches=[add, mul])
    print(a, m)
    
#To access graph in Tensorboard
#1. Open terminal.
#2. Check for graphs folder
#3. Run: tensornoard --logdir="./graphs" --port 6006
#4. Open browser and go to: http://localhost:6006/