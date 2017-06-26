
# A simple mixture density network attached to a feed forward network

# Written by 
# 	Luca Naterop
# 	Sandro Giacomuzzi
# as part of a semester project (AI music composition)
# at ETH Zuerich, 2017

# refer to README.md for usage


import matplotlib.pyplot as plt
import numpy as np
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
inputs = np.float32(np.random.uniform(size=N))
noise = np.float32(np.random.rand(N)*0.1 - 0.1)
targets = np.float32(inputs + 0.3*np.sin(2*np.pi*inputs) + noise)

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

# change roles of x and y
temp = inputs;
inputs = targets;
targets = temp;

# model
hidden = 5
L = 1 			# amount of standard outputs if there was no mdn
K = 6 			# amount of mixtures
outputs = (L+2)*K

x = tf.placeholder(dtype=tf.float32, shape=[None,1])										# (N, 1)
y = tf.placeholder(dtype=tf.float32, shape=[None,1])										# (N, 1)

W = tf.Variable(tf.random_normal([1,hidden], stddev=1.0, dtype=tf.float32))					
b = tf.Variable(tf.random_normal([1,hidden], stddev=1.0, dtype=tf.float32))
W_out = tf.Variable(tf.random_normal([hidden,outputs], stddev=1.0, dtype=tf.float32))
b_out = tf.Variable(tf.random_normal([outputs], stddev=1.0, dtype=tf.float32))
hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
y_out = tf.matmul(hidden_layer,W_out) + b_out												# (N, (L+2)K)

def getMixtureCoefficients(output):
	pi_k 		= tf.placeholder(dtype=tf.float32, shape=[None,K], name="mixparam")			# (N, K)
	sigma_k 	= tf.placeholder(dtype=tf.float32, shape=[None,K], name="mixparam")			# (N, K)
	mu_k 		= tf.placeholder(dtype=tf.float32, shape=[None,K], name="mixparam")			# (N, K)
	pi_k, sigma_k, mu_k = tf.split(1,3,output)												# (N, K)
	pi_k = tf.nn.softmax(pi_k)																# (N, K)
	sigma_k = tf.exp(sigma_k) 																# (N, K)
	return pi_k, sigma_k, mu_k
pi_k, sigma_k, mu_k = getMixtureCoefficients(y_out)											# (N, K)

def gaussian(y, mu_k, sigma_k):
	phi_k = -tf.div(tf.square(y-mu_k), 2*tf.square(sigma_k))		
	phi_k = tf.exp(phi_k)
	phi_k = tf.divide(phi_k, sigma_k)
	return phi_k

phi_k = gaussian(y, mu_k, sigma_k)
loss = tf.mul(phi_k,pi_k)
loss = tf.reduce_sum(loss, 1, keep_dims=True)
loss = tf.reduce_sum(-tf.log(loss))

# other branch of graph for predictions
max_indices = tf.argmax(pi_k, 1)

train_op = tf.train.AdamOptimizer(0.005).minimize(loss)

# launch session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
batch_size = 1000

# Fit the line.
epochs = 2000
for epoch in range(epochs):
	for i in range(0,N,batch_size):
		input = inputs[i:i+batch_size].reshape(batch_size,1)
		target = targets[i:i+batch_size].reshape(batch_size,1)
		_, cost = sess.run([train_op, loss],{x: input, y: target})
	print(' epoch = ' + str(epoch) + ', i = ' + str(i) + ' , loss = ' + str(cost*N/batch_size))

# sample
mu_i, maxima, pi_i, sigma_i = sess.run([mu_k, max_indices, pi_k, sigma_k], {x: inputs.reshape(N,1)})
maxima = np.array(maxima)
mu_i = np.array(mu_i)

def meanOfMaxProb(all_mu, max_indices):
	result = np.zeros(all_mu.shape[0])
	for i in range(all_mu.shape[0]):
		result[i] = all_mu[i, max_indices[i]]
	return result

def meanByProb(all_mu, pi_i):
	result = np.zeros(all_mu.shape[0])
	elements = np.linspace(0,K-1,K)
	for i in range(all_mu.shape[0]):
		index = np.random.choice(elements, 1, p=pi_i[i])[0]
		result[i] = all_mu[i,index]
	return result

def distByProb(all_mu, pi_i, sgima_i):
	result = np.zeros(all_mu.shape[0])
	elements = np.linspace(0,K-1,K)
	for i in range(all_mu.shape[0]):
		index = np.random.choice(elements, 1, p=pi_i[i])[0]
		result[i] = np.random.normal()*sigma_i[i, index] + all_mu[i,index]
	return result

predictions = distByProb(mu_i, pi_i, sigma_i)

# plot
plt.plot(inputs,targets,'ro', inputs, mu_i, 'go', inputs, predictions, 'bo', alpha=0.3)
plt.show()

sess.close()

