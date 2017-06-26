
# This is used to sample from the trained model

# Written by 
# 	Luca Naterop
# 	Sandro Giacomuzzi
# as part of a semester project (AI music composition)
# at ETH Zuerich, 2017

# refer to README.md for usage

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Util import create_batches
from Util import meanOfMaxProb
from Util import meanByProb
from Util import distByProb
from Util import printAsCSV
import numpy as np
import tensorflow as tf
from tabulate import tabulate
import sys
import os


# parse cmd args: checkpoint, N, layers, hidden, seed, sampler

# python sample.py checkpoint -amount 3000 -layers 3 -hidden 256 -K 10 -seed file.csv -sampler greedy

if len(sys.argv) <= 1:
	print('Error: sample.py expects at least a checkpoint as argument')

checkpoint = sys.argv[1]
checkpoint_path = checkpoint

if '-amount' in sys.argv:
	index = sys.argv.index('-amount')
	N = int(sys.argv[index+1])
else:
	N = 1000	# default

if '-layers' in sys.argv:
	index = sys.argv.index('-layers')
	hidden_layers = int(sys.argv[index+1])
else:
	print('Error: Please provide the amount of hidden layers using e.g. -layers 3')

if '-hidden' in sys.argv:
	index = sys.argv.index('-hidden')
	hidden_units = int(sys.argv[index+1])
else:
	print('Error: Please provide the amount of hidden units using e.g. -units 256')

if '-K' in sys.argv:
	index = sys.argv.index('-K')
	K = int(sys.argv[index+1])
else:
	print('Error: Please provide the amount of mixtures using .g. -K 7')

if '-seed' in sys.argv:
	index = sys.argv.index('-seed')
	seed_file = sys.argv[index+1]
	zero_seed = False
else:
	zero_seed = True

if '-L' in sys.argv:
	# it is the vector dimension, must be provided if theres no seed to take the dimension from
	index = sys.argv.index('-L')
	seed_file = sys.argv[index+1]
	L = int(sys.argv[index+1])
else: 
	L = 100

if '-sampler' in sys.argv:
	index = sys.argv.index('-sampler')
	sampler = sys.argv[index+1]
else:
	sampler = 'greedy'


# Reading data
# print('reading data..')
data = np.genfromtxt(seed_file, delimiter=',')
data_max = np.max(data)
data = data/data_max

if not zero_seed:
	L = data.shape[1]		# amount of standard outputs if there was no mdn

# print('--------------------------------- PARAMETERS ---------------------------------')
# print('N = ' + str(N) + ', L = ' + str(L) + ', K = ' + str(K) + 
# 	', hidden_units = ' + str(hidden_units) + ', hidden_layers = ' + str(hidden_layers))
# print('------------------------------------------------------------------------------')

# model
# print('creating the model...')
outputs = (L+2)*K

x = tf.placeholder(dtype=tf.float64, shape=[None,None,L])				# (B,T,L)
y = tf.placeholder(dtype=tf.float64, shape=[None,L])					# (B,L)

B_ = tf.shape(x)[0]
T = tf.shape(x)[1]

# init_state = tf.placeholder(tf.float64, [hidden_layers, 2, None, hidden_units])
# l = tf.unstack(init_state, axis=0)
# rnn_tuple_state = tuple( [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(hidden_layers)])

cell = tf.contrib.rnn.LSTMCell(hidden_units, use_peepholes=False)
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.7)
cell = tf.contrib.rnn.MultiRNNCell([cell] * hidden_layers)
output, state = tf.nn.dynamic_rnn(cell,x, dtype=tf.float64)				# output: (B,T,H), state: ([B,H],[B,H])
output = tf.transpose(output, [1,0,2])									# (T,B,H)
last = tf.gather(output, T-1)											# (B,H)

W_out = tf.Variable(tf.random_normal([hidden_units,outputs], stddev=0.001, dtype=tf.float64))
b_out = tf.Variable(tf.random_normal([outputs], stddev=0.001, dtype=tf.float64))
y_out = tf.matmul(last,W_out) + b_out									# (B,(L+2)K)

def getMixtureCoefficients(output):
	pi_k, sigma_k, mu_k = tf.split(output, [K,K,L*K], 1)									# 
	mu_k = tf.reshape(mu_k, [B_,K,L])														# (B,K,L)
	pi_k = tf.nn.softmax(pi_k)																# (B,K)
	sigma_k = tf.exp(sigma_k) 																# (B,K)
	return pi_k, sigma_k, mu_k
pi_k, sigma_k, mu_k = getMixtureCoefficients(y_out)											# (B,K)

def gaussian(y, mu_k, sigma_k):
	y = tf.reshape(y, [B_,1,L])
	norm = tf.reduce_sum(tf.square(y-mu_k),axis=2)	# sums over the L dimensions -> we get shape (B,K) again
	phi_k = -tf.div(norm, 2*tf.square(sigma_k))		
	phi_k = tf.exp(phi_k)
	phi_k = tf.divide(phi_k, sigma_k)
	return phi_k

phi_k = gaussian(y, mu_k, sigma_k)
loss = tf.multiply(phi_k,pi_k)
loss = tf.reduce_sum(loss, 1, keep_dims=True)
loss = tf.reduce_sum(-tf.log(loss))

# other branch of graph for predictions
max_indices = tf.argmax(tf.div(pi_k, sigma_k), 1)


# sample
def sample(Seed, amount):
	"synthesizes from the model"
	seed = np.copy(Seed)
	B = 1
	S = seed.shape[0]
	L = seed.shape[1]
	samples = np.zeros([S+amount, L])
	samples[0:S,:] = seed
	# State = np.zeros([hidden_layers,2,1,hidden_units])

	for i in range(0,amount):
		mu_i, maxima, pi_i, sigma_i, State = sess.run([mu_k, max_indices, pi_k, sigma_k, state], {x: np.expand_dims(seed, axis=0)})
		maxima 	= np.array(maxima)
		mu_i 	= np.array(mu_i)
		pi_i 	= np.array(pi_i)
		sigma_i = np.array(sigma_i)
		predictions = meanByProb(mu_i,pi_i,sigma_i)
		# seed = predictions
		seed[0:-1,:] = seed[1:,:]
		seed[-1,:] = predictions
		samples[S+i,:] = predictions
	samples = samples
	return samples

# launch session and load checkpoint
sess = tf.Session()
saver = tf.train.Saver(max_to_keep=1)
saver.restore(sess, checkpoint_path)

# sample and save
seed = data
# print('sample from the model')
samples = sample(seed, N)
# print('rescale samples')
samples = samples*data_max
# print('output in csv format')
printAsCSV(np.round(samples,3))
