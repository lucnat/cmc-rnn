
# Simple feed-forward neural network

# Written by 
# 	Luca Naterop
# 	Sandro Giacomuzzi
# as part of a semester project (AI music composition)
# at ETH Zuerich, 2017

# refer to README.md for usage

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

# generate training data
N = 1000
inputs = np.float32(np.random.uniform(size=N))
noise = np.float32(np.random.rand(N)*0.1 - 0.1)
targets = np.float32(inputs + 0.3*np.sin(2*np.pi*inputs) + noise)

# change roles of x and y
temp = inputs;
inputs = targets;
targets = temp;

# model
hidden = 10

x = tf.placeholder(dtype=tf.float32, shape=[None,1])
y = tf.placeholder(dtype=tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([1,hidden], stddev=1.0, dtype=tf.float32))
b = tf.Variable(tf.random_normal([1,hidden], stddev=1.0, dtype=tf.float32))
W_out = tf.Variable(tf.random_normal([hidden,1], stddev=1.0, dtype=tf.float32))
b_out = tf.Variable(tf.random_normal([1,1], stddev=1.0, dtype=tf.float32))
hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
y_out = tf.matmul(hidden_layer,W_out) + b_out
lossfunc = tf.nn.l2_loss(y_out-y);
train_op = tf.train.AdamOptimizer(0.1).minimize(lossfunc)

# launch session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 2000
for i in range(epochs):
  _, loss = sess.run([train_op, lossfunc] ,feed_dict={x: inputs.reshape(N,1), y: targets.reshape(N,1)})
  print('epoch = ' + str(i) + ', loss = ' + str(loss))

# sample
y_generated = sess.run([y_out], {x: inputs.reshape(N,1)})
y_generated = np.array(y_generated).reshape(1000)

# plot
plt.plot(inputs,targets,'ro', inputs,y_generated,'bo',alpha=0.3)
plt.show()

sess.close()

