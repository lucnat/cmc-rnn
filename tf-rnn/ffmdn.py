import numpy as np
from numpy import genfromtxt
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from batchloader import BatchLoader
import sys

# network parameters
inputDimension = 1
hiddenDimension = 20
outputDimension = 1

# generate training data
N = 1000
inputs = np.linspace(0,1,N)
noise = np.random.rand(N)*0.2 - 0.1
targets = inputs + 0.3*np.sin(2*np.pi*inputs) + noise

# shuffle
data = np.array([inputs, targets]).T
np.random.shuffle(data)
data = data.T
inputs = data[0]
targets = data[1]

# reverse roles of input and target
temp = inputs
inputs = targets
targets = temp

# model
x = tf.placeholder(tf.float32, [None, inputDimension])
W1 = tf.Variable(tf.random_normal([inputDimension, hiddenDimension]))
b1 = tf.Variable(tf.random_normal([hiddenDimension]))
h1 = tf.tanh(tf.matmul(x, W1) + b1)
W2 = tf.Variable(tf.random_normal([hiddenDimension, outputDimension]))
b2 = tf.Variable(tf.random_normal([outputDimension]))
prediction = tf.matmul(h1,W2) + b2

y = tf.placeholder(tf.float32, [None, inputDimension])

error = tf.reduce_mean(tf.square(y-prediction))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

batch_size = 10

def generate():
	# generate
	x_generated = np.linspace(0,1,50).reshape(50,1)
	y_generated = sess.run([prediction], {x: x_generated})
	y_generated = np.array(y_generated).reshape(50)
	plt.plot(inputs, targets, 'o', x_generated, y_generated)
	plt.show()

# Fit the line.
for epoch in range(1000):
	for i in range(0,N,batch_size):
		input = inputs[i:i+batch_size].reshape(batch_size,1)
		target = targets[i:i+batch_size].reshape(batch_size,1)
		_, loss = sess.run([train, error],{x: input, y: target})
		print(' epoch = ' + str(epoch) + ', i = ' + str(i) + ' , loss = ' + str(loss))

	if epoch % 100 == 0:
		generate()

