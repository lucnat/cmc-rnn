import numpy as np
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt

# generate training data
print("generating input...")
x = np.zeros([1000,10])
for i in range(1000):
	x[i][i%10] = 1

x = x.reshape([1000,10,1])

print("generating targets...")
y = np.zeros([1000,10])
for i in range(1000):
	y[i][(i+1)%10] = 1

train_size = 700

# split in training and test set
x_train = x[:train_size]
y_train = y[:train_size]
x_test = x[train_size:]
y_test = y[train_size:]

# model
data = tf.placeholder(tf.float32, [None, 10, 1])
target = tf.placeholder(tf.float32, [None, 10])

num_hidden = 100
num_layers = 20
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
output, state = tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.matmul(last, weight) + bias
loss =  tf.reduce_mean(tf.square(prediction-target))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(loss)

#execution
print("EXECUTE")
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
batch_size = 100
no_of_batches = int(len(x_train)/batch_size)
epochs = 1000

for i in range(epochs):
	ptr = 0
	for j in range(no_of_batches):
		inp, out = x_train[ptr:ptr+batch_size], y_train[ptr:ptr+batch_size]
		ptr+=batch_size
		sess.run(minimize,{data: inp, target: out})
		if j == 0:
			cost = sess.run(loss,{data: x_test, target: y_test})
			print("loss = ", cost)

		print("Epoch - ",str(i))

	inp = x_train[0:1]
	print(sess.run(prediction,{data: inp}))
sess.close()



