import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
from batchloader import BatchLoader
from tabulate import tabulate
import sys

# Parse command line arguments
csvfile = sys.argv[1]

# Reading data
print('reading data..')
data = np.genfromtxt(csvfile, delimiter=',')
L = data.shape[1]

# Parameters
L_in = L 		
L_out = L			# amount of standard outputs if there was no mdn
N = data.shape[0]	# amount of toy data set size and prediction size
hidden = 10			# amount of hidden layers in the network
K = 1 				# amount of mixtures
batch_size = N
epochs = 1000
learning_rate = 0.001

# # 2d data set for non-inverse problem
# print('generating dummy data')
# inputs = np.linspace(-3,3,N)
# targets1 = 0.1 + 0.3*inputs
# targets2 = 0.1 - 0.2*inputs

# # add second line for inverse problem
# for i in range(0,N):
# 	if i%2 == 0:
# 		targets1[i] = -0.2 - 0.2*inputs[i]
# 		targets2[i] = -0.1 + 0.1*inputs[i]

# targets = [targets1, targets2]
# targets = np.transpose(targets)

# model
print('creating the model...')
outputs = (L_out+2)*K

x = tf.placeholder(dtype=tf.float32, shape=[None,L_in])											# (N,L)
y = tf.placeholder(dtype=tf.float32, shape=[None,L_out])										# (N,1)

batchDim = tf.shape(x)[0]

W = tf.Variable(tf.random_normal([L_in,hidden], stddev=0.1, dtype=tf.float32))		
b = tf.Variable(tf.random_normal([hidden], stddev=0.1, dtype=tf.float32))
W_out = tf.Variable(tf.random_normal([hidden,outputs], stddev=0.1, dtype=tf.float32))
b_out = tf.Variable(tf.random_normal([outputs], stddev=0.1, dtype=tf.float32))
hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
y_out = tf.matmul(hidden_layer,W_out) + b_out												# (N,(L+2)K)

def getMixtureCoefficients(output):
	pi_k, sigma_k, mu_k = tf.split(output, [K,K,L_out*K], 1)									# 
	mu_k = tf.reshape(mu_k, [batchDim,K,L_out])														# (N,K,L)
	pi_k = tf.nn.softmax(pi_k)																# (N,K)
	sigma_k = tf.exp(sigma_k) 																# (N,K)
	return pi_k, sigma_k, mu_k
pi_k, sigma_k, mu_k = getMixtureCoefficients(y_out)											# (N,K)

def gaussian(y, mu_k, sigma_k):
	y = tf.reshape(y, [batchDim,1,L_out])
	norm = tf.reduce_sum(tf.square(y-mu_k),axis=2)	# sums over the L dimensions -> we get shape (N,K) again
	phi_k = -tf.div(norm, 2*tf.square(sigma_k))		
	phi_k = tf.exp(phi_k)
	phi_k = tf.divide(phi_k, sigma_k)
	return phi_k

phi_k = gaussian(y, mu_k, sigma_k)
loss = tf.multiply(phi_k,pi_k)
loss = tf.reduce_sum(loss, 1, keep_dims=True)
loss = tf.reduce_sum(-tf.log(loss))

# other branch of graph for predictions
max_indices = tf.argmax(pi_k, 1)

train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# launch session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
print('starting training...')

batchLoader = BatchLoader(data,batch_size)

for epoch in range(epochs):
	isLastBatch = False
	i = 0
	while not isLastBatch:
		i += 1
		inputs, targets, isLastBatch = batchLoader.nextRNNBatch()
		_, cost = sess.run([train_op, loss],{x: inputs, y: targets})
		print(' epoch = ' + str(epoch) + ', i = ' + str(i) + ' , loss = ' + str(cost*N/batch_size))

print('sampling from the model....')
mu_i, maxima, pi_i, sigma_i = sess.run([mu_k, max_indices, pi_k, sigma_k], {x: data})
maxima 	= np.array(maxima)
mu_i 	= np.array(mu_i)
pi_i 	= np.array(pi_i)
sigma_i = np.array(sigma_i)

def meanOfMaxProb(all_mu, max_indices):
	result = np.zeros([N,L_out])
	for i in range(N):
		result[i, :] = all_mu[i, max_indices[i], :]
	return result

def meanByProb(all_mu, pi_i):
	result = np.zeros([N,L_out])
	elements = np.linspace(0,K-1,K)
	for i in range(all_mu.shape[0]):
		index = np.random.choice(elements, 1, p=pi_i[i])[0]
		result[i] = all_mu[i,index]
	return result

def distByProb(all_mu, pi_i, sgima_i):
	result = np.zeros([N,L_out])
	elements = np.linspace(0,K-1,K)
	for i in range(all_mu.shape[0]):
		index = np.random.choice(elements, 1, p=pi_i[i])[0]
		result[i] = np.random.normal()*sigma_i[i, index] + all_mu[i,index]
	return result

# sample
def sample(amount):
	"synthesizes from the model"
	# let's sample a bit here to see whats going on
	print('sampling...')
	samples = np.zeros([amount,N])
	inp = data[0:1]
	for k in range(amount):
		samples[k] = inp.reshape([N])
		inp = sess.run(prediction,{x: inp})
	return samples

predictions = meanOfMaxProb(mu_i, maxima)
print(tabulate(predictions[:100]))

# plot
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot(inputs, targets1, targets2, '.')
# ax.plot(inputs, np.transpose(predictions)[0], np.transpose(predictions)[1], '.')
# plt.show()

sess.close()
