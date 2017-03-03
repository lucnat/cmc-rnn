import numpy as np
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from batchloader import BatchLoader
from tabulate import tabulate
import sys

# Parse command line arguments
csvfile = sys.argv[1]

# Reading data
print('reading data..')
x = np.genfromtxt(csvfile, delimiter=',')

# Hyper Parameters
N = x.shape[1] 		# input dimension
num_hidden = 10		# amount of hidden units
num_layers = 1		# amount of hidden layers
lr = 0.01 			# learning reate
train_size = 400	# size of training set
sequence_length = 20

print('------------- PARAMETERS --------------')
print('N = ' + str(N) + ', num_hidden = ' + str(num_hidden) + ', num_layers = ' + str(num_layers) + ', lr = ' + str(lr) + ', train_size = ' + str(train_size) + ', m = ' + str(x.shape[0]))
print('--------------------------------------')

print('creating model..')
# split in training and test set
x_train = x[:train_size]
x_test = x[train_size:]
y_test = x_test[1:]
x_test = x_test[:x_test.shape[0]-1]

# model
data = tf.placeholder(tf.float32, [None,sequence_length, N])
target = tf.placeholder(tf.float32, [None, N])
cell = tf.contrib.rnn.BasicRNNCell(num_hidden)
cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
output, state = tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.matmul(last, weight) + bias
loss =  tf.reduce_mean(tf.square(prediction-target))
optimizer = tf.train.AdamOptimizer(lr)
minimize = optimizer.minimize(loss)

# run the session
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

def sample(seed,amount):
	"synthesizes from the model"
	# let's sample a bit here to see whats going on
	print('sampling...')
	S = seed.shape[0] 				# seed length
	samples = np.zeros([S+amount,N])
	samples[0:S] = seed

	pred = seed
	for k in range(0,amount):
		pred = sess.run(prediction, {data: pred})
		samples[k+S,:] = pred[-1,:]
	return samples


def test():
	"computes the loss on the test set"
	# let's measure performance on test set
	global x_test
	x_test = x_test.reshape([x_test.shape[0],x_test.shape[1],1]) # make it (?, N, 1)
	test_loss = sess.run(loss,{data: x_test, target: y_test})
	print("test_loss = " + str(test_loss))


def train():
	"trains the model"
	print('start training...')
	batch_size = 800
	batchLoader = BatchLoader(x_train,batch_size)

	no_of_batches = int(len(x_train)/batch_size)
	epochs = 1000
	cost = 10

	for epoch in range(epochs):
		isLastBatch = False

		inputs = x[0:-1,:]
		targets = x[1:,:]
		inputs = np.reshape(inputs, [40,20,N])
		targets = np.reshape(targets, [800,N])
		_, cost = sess.run([minimize, loss],{data: inputs, target: targets})

		print('epoch = ' + str(epoch) + ', cost = ' + str(cost))

		# if i%10 == 0:
			# samples = sample(50)
			# filename = 'e'+str(epoch)+'loss'+str(cost)+'.csv'
			# print('writing ' + filename + '...')
			# np.savetxt(filename,samples,delimiter=",")
			# print('done')

		# if epoch%10 == 0:
			# filename = 'e'+str(epoch)+'loss'+str(cost)+'.csv'
			# print('writing ' + filename + '...')
			# np.savetxt(filename,samples,delimiter=",")
			# print('done')

train()

seed = x[0:10,:]
samples = sample(seed, 20)
print(tabulate(samples))

sess.close()


