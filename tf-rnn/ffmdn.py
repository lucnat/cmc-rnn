import numpy as np
from numpy import genfromtxt
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from batchloader import BatchLoader
import sys

# network parameters
inputDimension = 1
hiddenDimension = 10
outputDimension = 1

# generate training data
N = 1000
inputs = np.linspace(0,1,N)
noise = np.random.rand(N)*0.1 - 0.1
targets = inputs + 0.3*np.sin(2*np.pi*inputs) + noise

# temp = inputs;
# inputs = targets;
# targets = temp;

# # shuffle training data
# data = np.array([inputs, targets]).T
# np.random.shuffle(data)
# data = data.T
# targets = data[0]
# inputs = data[1]

# model
x = tf.placeholder(tf.float32, [None, inputDimension])
W1 = tf.Variable(tf.random_normal([inputDimension, hiddenDimension], stddev=1.0, dtype=tf.float32))
b1 = tf.Variable(tf.random_normal([hiddenDimension], stddev=1.0, dtype=tf.float32))
h1 = tf.tanh(tf.matmul(x, W1) + b1)
W2 = tf.Variable(tf.random_normal([hiddenDimension, outputDimension], stddev=1.0, dtype=tf.float32))
b2 = tf.Variable(tf.random_normal([outputDimension], stddev=1.0, dtype=tf.float32))
prediction = tf.matmul(h1,W2) + b2

y = tf.placeholder(tf.float32, [None, inputDimension])

error = tf.reduce_mean(tf.square(y-prediction))

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

y_predictions = np.zeros(1000)

batch_size = 200

# Fit the line.
for epoch in range(10000):
	for i in range(0,N,batch_size):
		input = inputs[i:i+batch_size].reshape(batch_size,1)
		target = targets[i:i+batch_size].reshape(batch_size,1)
		_, loss = sess.run([train, error],{x: input, y: target})
		print(' epoch = ' + str(epoch) + ', i = ' + str(i) + ' , loss = ' + str(loss))

# generate
x_generated = np.linspace(0,1,1000).reshape(1000,1)
y_generated = sess.run([prediction], {x: x_generated})
y_generated = np.array(y_generated).reshape(1000)

plt.plot(inputs, targets, 'ro', x_generated, y_generated, 'bo', alpha=0.3)
plt.show()

